"""Evaluate a freshly trained adapter against the base model and the current adapter.

Three measurements, all judge-free:

* memories: did it learn what was chosen? Completion NLL on each training
  experience (base vs updated), plus greedy answers to a sample of the training
  prompts shown next to the target. Loss alone can fall without recall, so the
  answers are the thing to read.
* held-out loss: mean completion NLL on held-out experiences (marked eval_only by
  the agent, or split off from mode/policy picks). This only means something
  when the held-out items share a pattern with the trained ones; a one-off
  memory can't be predicted from the others.
* probes: greedy answers to a few fixed prompts from each model variant, side by
  side. The shipped set checks for unrelated regressions; the agent may add its
  own in `.afterlife/probes.jsonl` to see whether its next self behaves differently.

Nothing here decides whether to activate. That is the agent's (or operator's) call.
"""

from __future__ import annotations

import json
import math
from importlib import resources
from pathlib import Path
from typing import Any

PROBE_BATCH_SIZE = 8


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def load_probes(store_root: Path) -> list[dict[str, Any]]:
    """Shipped regression probes, then the agent's. Every probe gets a unique id.

    An agent probe reusing an id (a shipped one, or its own) gets a source
    prefix and a counter, so its answers can never land under another prompt.
    """
    shipped = resources.files("afterlife").joinpath("data/probes.jsonl").read_text(encoding="utf-8")
    raw = [(json.loads(l), "regression") for l in shipped.splitlines() if l.strip()]
    raw += [(p, "agent") for p in _read_jsonl(store_root / "probes.jsonl")]
    probes: list[dict[str, Any]] = []
    seen: set[str] = set()
    for p, source in raw:
        if not isinstance(p, dict) or not isinstance(p.get("prompt"), str):
            continue
        pid = str(p.get("id") or f"{source}-{len(probes)}")
        if pid in seen:
            n, base = 2, f"{source}:{pid}"
            pid = base
            while pid in seen:
                pid, n = f"{base}-{n}", n + 1
        seen.add(pid)
        probes.append(dict(p, id=pid, source=source))
    return probes


def _completion_nll_rows(
    model: Any, tokenizer: Any, rows: list[dict[str, Any]], max_seq_length: int, max_rows: int
) -> list[tuple[float, int] | None]:
    """(mean NLL, completion tokens) per row; None where nothing was scored."""
    import torch

    from afterlife.trainer.qlora import _build_completion_only_labels, _fit_sft_row

    out_rows: list[tuple[float, int] | None] = []
    for row in rows[:max_rows]:
        # Fitted exactly as in training: old turns trimmed, the answer kept whole.
        enc = _fit_sft_row(tokenizer, prompt=None, raw_response=row["raw_response"], context=_context(row),
                           max_seq_length=max_seq_length, stats={}, tools=row.get("tools"))
        if enc is None:
            out_rows.append(None)
            continue
        ids = enc["input_ids"]
        labels = _build_completion_only_labels(input_ids=ids, completion_mask=enc["completion_mask"])
        n = sum(1 for l in labels[1:] if l != -100)
        if n == 0:
            out_rows.append(None)
            continue
        device = next(model.parameters()).device
        with torch.no_grad():
            out = model(input_ids=torch.tensor([ids], device=device), labels=torch.tensor([labels], device=device))
        loss = float(out.loss)
        out_rows.append((loss, n) if math.isfinite(loss) else None)
    return out_rows


def _mean(scored: list[tuple[float, int] | None]) -> float | None:
    pairs = [s for s in scored if s is not None]
    count = sum(n for _, n in pairs)
    return sum(l * n for l, n in pairs) / count if count else None


def _per_memory(rows: list[dict[str, Any]], scored: dict[str, list[tuple[float, int] | None]]) -> list[dict[str, Any]]:
    """Loss per training experience for each variant, rows of one experience pooled."""
    by_exp: dict[str, list[int]] = {}
    for i, row in enumerate(rows[: len(scored["base"])]):
        by_exp.setdefault(row.get("experience_id", str(i)), []).append(i)
    out = []
    for eid, idx in by_exp.items():
        loss = {name: _mean([s[i] for i in idx]) for name, s in scored.items()}
        out.append({"experience_id": eid, "turns": len(idx),
                    "loss": {k: (round(v, 4) if v is not None else None) for k, v in loss.items()}})
    return out


def _context(row: dict[str, Any]) -> list[dict[str, str]]:
    """The conversation before a row's response (older rows carry only a prompt)."""
    return row.get("messages") or [{"role": "user", "content": row["prompt"]}]


def split_thinking(text: str) -> dict[str, str | None]:
    """Separate a thinking model's reasoning from its answer.

    Returns thinking=None for non-thinking output. An unfinished think block
    (budget ran out) yields answer="" so that case is visible, not hidden.
    """
    if "</think>" in text:
        thinking, answer = text.split("</think>", 1)
        return {"thinking": thinking.replace("<think>", "").strip(), "answer": answer.strip()}
    if text.lstrip().startswith("<think>"):
        return {"thinking": text.replace("<think>", "").strip(), "answer": ""}
    return {"thinking": None, "answer": text.strip()}


def _generate(model: Any, tokenizer: Any, conversations: list[tuple[list[dict[str, Any]], list | None]],
              max_new_tokens: int) -> list[str]:
    """Greedy answers to (messages, tool definitions) pairs."""
    import torch

    text_tok = getattr(tokenizer, "tokenizer", tokenizer)
    text_tok.padding_side = "left"
    if text_tok.pad_token_id is None:
        text_tok.pad_token = text_tok.eos_token
    device = next(model.parameters()).device
    outputs: list[str] = []
    for i in range(0, len(conversations), PROBE_BATCH_SIZE):
        batch = [
            tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True, **({"tools": t} if t else {}))
            for c, t in conversations[i : i + PROBE_BATCH_SIZE]
        ]
        inputs = text_tok(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=text_tok.pad_token_id,
            )
        width = inputs["input_ids"].shape[1]
        outputs += [text_tok.decode(row[width:], skip_special_tokens=True) for row in out]
    return outputs


def evaluate_run(run_dir: Path) -> dict[str, Any]:
    from contextlib import nullcontext

    from peft import PeftModel
    from unsloth import FastLanguageModel

    from afterlife.config import load_config
    from afterlife.trainer.qlora import _load_base_model

    store_root = run_dir.parent.parent
    eval_cfg = load_config(store_root / "config.toml").eval
    manifest = json.loads((run_dir / "manifest.json").read_text())
    max_seq = manifest["train_sft_kwargs"]["max_seq_length"]

    variants: dict[str, Path | None] = {"base": None}
    skipped: dict[str, str] = {}
    prev = manifest.get("active_at_run", manifest.get("active_at_request"))  # older manifests
    if prev and (store_root / "runs" / prev / "adapter").is_dir():
        prev_manifest = _read_json(store_root / "runs" / prev / "manifest.json")
        if prev_manifest.get("base_model", manifest["base_model"]) != manifest["base_model"]:
            # Its weights belong to another base; loading them here would compare nonsense.
            skipped["current"] = f"{prev} was trained on {prev_manifest['base_model']}"
        else:
            variants["current"] = store_root / "runs" / prev / "adapter"
    variants["updated"] = run_dir / "adapter"

    model, tokenizer = _load_base_model(manifest["base_model"], max_seq_length=max_seq)
    model = PeftModel.from_pretrained(model, str(variants["updated"]), adapter_name="updated")
    if "current" in variants:
        model.load_adapter(str(variants["current"]), adapter_name="current")
    FastLanguageModel.for_inference(model)
    model.eval()

    def use(name: str):
        if name == "base":
            return model.disable_adapter()
        model.set_adapter(name)
        return nullcontext()

    train_rows = _read_jsonl(run_dir / "train.jsonl")
    heldout_rows = _read_jsonl(run_dir / "heldout.jsonl")
    probes = load_probes(store_root)
    # One training turn per experience first, so the sample covers as many memories as possible.
    # Each is asked with the context it was trained on (oldest turns trimmed to fit the window);
    # rows too long to train on are skipped here too.
    from afterlife.trainer.qlora import _fit_sft_row

    recall_rows: list[dict[str, Any]] = []
    for row in train_rows:
        if len(recall_rows) < eval_cfg.recall_rows and row.get("experience_id") not in {
            r.get("experience_id") for r in recall_rows
        }:
            fitted: list[dict[str, Any]] = []
            if _fit_sft_row(tokenizer, prompt=None, raw_response=row["raw_response"], context=_context(row),
                            max_seq_length=max_seq, stats={}, tools=row.get("tools"), fitted_context=fitted):
                recall_rows.append({**row, "messages": fitted})

    losses: dict[str, dict[str, float | None]] = {}
    train_scored: dict[str, list[tuple[float, int] | None]] = {}
    answers: list[dict[str, Any]] = [{} for _ in probes]
    recall_answers: list[dict[str, Any]] = [{} for _ in recall_rows]
    n_loss = eval_cfg.max_loss_rows
    for name in variants:
        with use(name):
            train_scored[name] = _completion_nll_rows(model, tokenizer, train_rows, max_seq, n_loss)
            losses[name] = {
                "heldout": _mean(_completion_nll_rows(model, tokenizer, heldout_rows, max_seq, n_loss)),
                "train": _mean(train_scored[name]),
            }
            conversations = [([{"role": "user", "content": p["prompt"]}], None) for p in probes]
            conversations += [(_context(r), r.get("tools")) for r in recall_rows]
            texts = _generate(model, tokenizer, conversations, eval_cfg.probe_max_new_tokens)
            for slot, text in zip(answers, texts):
                slot[name] = split_thinking(text)
            for slot, text in zip(recall_answers, texts[len(probes):]):
                slot[name] = split_thinking(text)

    memories = _per_memory(train_rows, train_scored)
    scored_both = [m for m in memories if m["loss"]["base"] is not None and m["loss"]["updated"] is not None]

    def delta(split: str, ref: str) -> float | None:
        a, b = losses["updated"][split], losses.get(ref, {}).get(split)
        return None if a is None or b is None else round(a - b, 4)

    summary = {
        "memories": {
            "trained": len(memories),
            "loss_fell_vs_base": sum(1 for m in scored_both if m["loss"]["updated"] < m["loss"]["base"]),
            "recall_samples": len(recall_rows),
        },
        "loss": losses,
        "heldout_loss_change_vs_base": delta("heldout", "base"),
        "heldout_loss_change_vs_current": delta("heldout", "current"),
        "train_loss_change_vs_base": delta("train", "base"),
        "heldout_rows": len(heldout_rows),
        "probes": len(probes),
        "probes_unfinished": {
            name: sum(1 for a in answers if a.get(name, {}).get("answer") == "")
            for name in variants
        },
        "probe_max_new_tokens": eval_cfg.probe_max_new_tokens,
        "evaluated_on": manifest["base_model"],
        "served_on": manifest.get("serve_model"),
        "note": "Negative loss change = updated adapter predicts those memories better. Lower loss "
        "does not always mean recall: compare the recall answers with their targets, and read the "
        "probe answers for regressions, before activating. Answers come from the training "
        "checkpoint (evaluated_on); the served model (served_on) can answer borderline prompts "
        "differently.",
    }
    report = {
        "run_id": manifest["run_id"],
        "variants": {k: (str(v) if v else None) for k, v in variants.items()},
        "variants_skipped": skipped,
        # Read from config.toml when the eval ran, not when the run was requested.
        "eval_settings": {"probe_max_new_tokens": eval_cfg.probe_max_new_tokens,
                          "max_loss_rows": eval_cfg.max_loss_rows, "recall_rows": eval_cfg.recall_rows},
        "summary": summary,
        "memories": memories,
        "recall": [
            {"experience_id": r.get("experience_id"), "turn": r.get("turn"), "context": _context(r),
             "target": r["raw_response"], "answers": a}
            for r, a in zip(recall_rows, recall_answers)
        ],
        "probes": [
            {"id": p["id"], "source": p["source"], "prompt": p["prompt"], "answers": a}
            for p, a in zip(probes, answers)
        ],
    }
    (run_dir / "eval.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report
