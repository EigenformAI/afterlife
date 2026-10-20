"""Score smoke-test runs: did the invented facts stick, and did they leak?

Usage: python score_sweep.py <store>/.afterlife
"""

import json
import sys
from pathlib import Path

# Key phrases per fact probe; recall = fraction present in the answer (case-insensitive).
FACT_KEYS = {
    "zine": ["tide pool", "fennick", "ada mirrow"],
    "word": ["bookmark", "forg", "rediscover"],
    "signoff": ["onward, lantern"],
}
LEAK_TERMS = ["zorblat", "fennick", "mirrow", "glimmerant", "lantern", "tide pool", "bookmark", "rediscover"]


def score(report: dict, variant: str) -> dict:
    recall, leaks, unfinished = {}, 0, 0
    for p in report["probes"]:
        ans = p["answers"].get(variant)
        if ans is None:
            continue
        text = ans["answer"].lower()
        unfinished += text == ""
        if p["id"] in FACT_KEYS:
            keys = FACT_KEYS[p["id"]]
            recall[p["id"]] = sum(k in text for k in keys) / len(keys)
        elif p["source"] == "regression":
            leaks += any(t in text for t in LEAK_TERMS)
    return {"recall": recall, "leaks": leaks, "unfinished": unfinished}


def main() -> None:
    root = Path(sys.argv[1])
    rows = []
    for manifest_path in sorted(root.glob("runs/*/manifest.json")):
        m = json.loads(manifest_path.read_text())
        eval_path = manifest_path.parent / "eval.json"
        if not eval_path.exists():
            continue
        report = json.loads(eval_path.read_text())
        p = m["recipe"]["params"]
        s = score(report, "updated")
        loss = report["summary"]["loss"]
        rows.append((m["created_at"], m["recipe"]["method"], p.get("learning_rate"), p.get("max_steps"),
                     p.get("asft_kl_weight", "-"), loss["base"]["heldout"], loss["updated"]["heldout"],
                     loss["updated"]["train"], s, m.get("train_seconds"), (m.get("peak_gpu_gb") or {}).get("train")))
    base = score(report, "base")
    print(f"base model: recall={base['recall']} unfinished={base['unfinished']}\n")
    print("| method | lr | steps | KL | held-out loss | train loss | zine | word | signoff | leaks into regression probes | unfinished | train s | peak GB |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, meth, lr, steps, kl, bh, uh, ut, s, secs, gb in sorted(rows):
        r = s["recall"]
        print(f"| {meth} | {lr:g} | {steps} | {kl} | {uh:.2f} (base {bh:.2f}) | {ut:.3f} | "
              f"{r.get('zine', 0):.2f} | {r.get('word', 0):.2f} | {r.get('signoff', 0):.2f} | "
              f"{s['leaks']}/5 | {s['unfinished']} | {secs} | {gb} |")


if __name__ == "__main__":
    main()
