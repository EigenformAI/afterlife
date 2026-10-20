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

MAX_LOSS_ROWS = 200
PROBE_MAX_NEW_TOKENS = 256


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_probes(store_root: Path) -> list[dict[str, Any]]:
    shipped = resources.files("afterlife").joinpath("data/probes.jsonl").read_text(encoding="utf-8")
    probes = [dict(json.loads(l), source="regression") for l in shipped.splitlines() if l.strip()]
    probes += [dict(p, source="agent") for p in _read_jsonl(store_root / "probes.jsonl")]
    return [p for p in probes if isinstance(p.get("prompt"), str)]


def _mean_completion_nll(model: Any, tokenizer: Any, rows: list[dict[str, Any]], max_seq_length: int) -> float | None:
    import torch

    from afterlife.trainer.qlora import _build_completion_only_labels, _build_query_response_sft_row

    total, count = 0.0, 0
    for row in rows[:MAX_LOSS_ROWS]:
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


def _generate(model: Any, tokenizer: Any, prompt: str) -> str:
    import torch

    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    text_tok = getattr(tokenizer, "tokenizer", tokenizer)
    inputs = text_tok(text, return_tensors="pt").to(next(model.parameters()).device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=PROBE_MAX_NEW_TOKENS, do_sample=False)
    return text_tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def evaluate_run(run_dir: Path) -> dict[str, Any]:
    from contextlib import nullcontext

    from peft import PeftModel
    from unsloth import FastLanguageModel

    from afterlife.trainer.qlora import _load_base_model

    store_root = run_dir.parent.parent
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
    answers: dict[str, dict[str, str]] = {p.get("id", str(i)): {} for i, p in enumerate(probes)}
    for name in variants:
        with use(name):
            losses[name] = {
                "heldout": _mean_completion_nll(model, tokenizer, heldout_rows, max_seq),
                "train": _mean_completion_nll(model, tokenizer, train_rows, max_seq),
            }
            for i, p in enumerate(probes):
                answers[p.get("id", str(i))][name] = _generate(model, tokenizer, p["prompt"])

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
