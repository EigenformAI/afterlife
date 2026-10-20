"""Evaluate a freshly trained adapter against the base model and the current adapter.

Two measurements, both cheap and judge-free:

* held-out loss: mean completion NLL on experiences held out by session. Lower
  means the adapter predicts the agent's own selected-but-unseen memories better.
  Also reported on the training rows, so overfitting is visible.
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


def load_probes(store_root: Path) -> list[dict[str, Any]]:
    shipped = resources.files("afterlife").joinpath("data/probes.jsonl").read_text(encoding="utf-8")
    probes = [dict(json.loads(l), source="regression") for l in shipped.splitlines() if l.strip()]
    probes += [dict(p, source="agent") for p in _read_jsonl(store_root / "probes.jsonl")]
    return [p for p in probes if isinstance(p.get("prompt"), str)]


def _mean_completion_nll(
    model: Any, tokenizer: Any, rows: list[dict[str, Any]], max_seq_length: int, max_rows: int
) -> float | None:
    import torch

    from afterlife.trainer.qlora import _build_completion_only_labels, _build_query_response_sft_row

    total, count = 0.0, 0
    for row in rows[:max_rows]:
        enc = _build_query_response_sft_row(tokenizer, prompt=row["prompt"], raw_response=row["raw_response"])
        ids = enc["input_ids"][:max_seq_length]
        labels = _build_completion_only_labels(input_ids=ids, completion_mask=enc["completion_mask"][:max_seq_length])
        n = sum(1 for l in labels[1:] if l != -100)
        if n == 0:
            continue
        device = next(model.parameters()).device
        with torch.no_grad():
            out = model(input_ids=torch.tensor([ids], device=device), labels=torch.tensor([labels], device=device))
        total += float(out.loss) * n
        count += n
    return total / count if count else None


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


def _generate(model: Any, tokenizer: Any, prompts: list[str], max_new_tokens: int) -> list[str]:
    import torch

    text_tok = getattr(tokenizer, "tokenizer", tokenizer)
    text_tok.padding_side = "left"
    if text_tok.pad_token_id is None:
        text_tok.pad_token = text_tok.eos_token
    device = next(model.parameters()).device
    outputs: list[str] = []
    for i in range(0, len(prompts), PROBE_BATCH_SIZE):
        batch = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
            )
            for p in prompts[i : i + PROBE_BATCH_SIZE]
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
    prev = manifest.get("active_at_request")
    if prev and (store_root / "runs" / prev / "adapter").is_dir():
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

    losses: dict[str, dict[str, float | None]] = {}
    answers: dict[str, dict[str, Any]] = {p.get("id", str(i)): {} for i, p in enumerate(probes)}
    n_loss = eval_cfg.max_loss_rows
    for name in variants:
        with use(name):
            losses[name] = {
                "heldout": _mean_completion_nll(model, tokenizer, heldout_rows, max_seq, n_loss),
                "train": _mean_completion_nll(model, tokenizer, train_rows, max_seq, n_loss),
            }
            texts = _generate(model, tokenizer, [p["prompt"] for p in probes], eval_cfg.probe_max_new_tokens)
            for i, (p, text) in enumerate(zip(probes, texts)):
                answers[p.get("id", str(i))][name] = split_thinking(text)

    for split in losses.values():
        for k, v in split.items():
            if v is not None and not math.isfinite(v):
                split[k] = None

    def delta(split: str, ref: str) -> float | None:
        a, b = losses["updated"][split], losses.get(ref, {}).get(split)
        return None if a is None or b is None else round(a - b, 4)

    summary = {
        "loss": losses,
        "heldout_loss_change_vs_base": delta("heldout", "base"),
        "heldout_loss_change_vs_current": delta("heldout", "current"),
        "train_loss_change_vs_base": delta("train", "base"),
        "heldout_rows": len(heldout_rows),
        "probes": len(probes),
        "probes_unfinished": {
            name: sum(1 for a in answers.values() if a.get(name, {}).get("answer") == "")
            for name in variants
        },
        "probe_max_new_tokens": eval_cfg.probe_max_new_tokens,
        "note": "Negative loss change = updated adapter predicts those memories better. "
        "Read the probe answers before activating; loss does not show regressions on its own.",
    }
    report = {
        "run_id": manifest["run_id"],
        "variants": {k: (str(v) if v else None) for k, v in variants.items()},
        "summary": summary,
        "probes": [
            {"id": p.get("id", str(i)), "source": p["source"], "prompt": p["prompt"],
             "answers": answers[p.get("id", str(i))]}
            for i, p in enumerate(probes)
        ],
    }
    (run_dir / "eval.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report
