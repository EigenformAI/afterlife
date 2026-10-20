# GPU smoke test: can SFT / DFT / ASFT learn new facts?

Run 2026-10-20 on one RTX A6000 (48 GB). Base `unsloth/Qwen3-8B-unsloth-bnb-4bit`,
LoRA rank 32 / alpha 32, batch 2 x grad-accum 4, linear schedule, 5 warmup steps,
seed 42, max_seq_length 4096, fixed completion mask (commit 216771c).

Data: `make_dataset.py`. Three invented facts, each asked 8 ways, plus 4
ordinary web tips: 24 train / 4 held-out rows, split by session. Probes:
3 fact questions (`probes.jsonl`) + 5 shipped regression prompts, greedy.
Scored with `score_sweep.py`: recall = fraction of key phrases present in
the probe answer; leaks = regression answers containing any fact term.

Probe budget was 2048 tokens for the first three rows, 1024 for the rest.
An answer cut off mid-thought scores 0 recall (see the unfinished column).

| method | lr | steps | KL | held-out loss | train loss | zine | word | signoff | leaks into regression probes | unfinished | train s | peak GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sft | 0.0002 | 40 | - | 0.00 (base 5.86) | 0.000 | 1.00 | 1.00 | 0.00 | 1/5 | 1 | 81.8 | 7.96 |
| dft | 0.0002 | 40 | - | 7.41 (base 5.86) | 6.037 | 0.00 | 0.33 | 0.00 | 0/5 | 1 | 85.9 | 7.95 |
| asft | 0.0002 | 40 | 0.1 | 4.21 (base 5.86) | 3.559 | 0.00 | 0.33 | 0.00 | 0/5 | 1 | 98.4 | 8.39 |
| sft | 5e-05 | 40 | - | 1.55 (base 5.86) | 1.233 | 0.00 | 0.00 | 0.00 | 0/5 | 2 | 81.8 | 7.96 |
| dft | 0.0005 | 40 | - | 6.91 (base 5.86) | 6.959 | 0.00 | 0.00 | 0.00 | 0/5 | 3 | 86.5 | 7.95 |
| dft | 0.0002 | 150 | - | 7.39 (base 5.86) | 6.059 | 0.00 | 0.00 | 0.00 | 0/5 | 3 | 221.9 | 7.96 |
| dft | 0.0005 | 150 | - | 7.17 (base 5.86) | 6.795 | 0.00 | 0.00 | 0.00 | 0/5 | 8 | 224.3 | 7.96 |
| asft | 0.0005 | 40 | 0.1 | 3.67 (base 5.86) | 3.018 | 0.00 | 0.00 | 0.00 | 0/5 | 2 | 99.1 | 8.39 |
| asft | 0.0002 | 150 | 0.1 | 3.67 (base 5.86) | 3.011 | 0.00 | 0.33 | 0.00 | 0/5 | 2 | 271.9 | 8.4 |
| asft | 0.0005 | 150 | 0.1 | 2.66 (base 5.86) | 2.401 | 0.00 | 0.33 | 0.00 | 0/5 | 1 | 258.8 | 8.4 |
| asft | 0.0005 | 150 | 0.02 | 2.02 (base 5.86) | 1.740 | 0.00 | 0.00 | 0.00 | 0/5 | 2 | 270.6 | 8.4 |

## Reading

- **SFT** learns new facts abruptly. At 5e-5 x 40 steps it reaches held-out loss
  1.55 but recalls nothing. At 2e-4 x 40 it recalls everything word for word,
  and the facts start leaking into unrelated answers ("each answer a bookmark
  waiting to be rediscovered").
- **DFT** learns the *shape* of the answers but not the new content:
  "Noted. I'll keep it brief. —D" instead of "— onward, Lantern"; "Fennville,
  Michigan" instead of Fennick Bay. Training loss rises in every DFT run. At
  5e-4 x 150 steps the model degenerates (repetition loops, 17 x 23 = 390).
  This fits the objective: tokens are weighted by the model's own
  probability, so near-zero-probability new facts get almost no gradient.
- **ASFT** is stable at every setting (arithmetic intact, no leaks), and
  held-out loss keeps falling (5.86 -> 2.02). Within 150 steps it still never
  states the facts at greedy decoding.

These match NSL2's finding that DFT/ASFT are more reliable than SFT, which was
measured on self-generated trajectories, i.e. content the model already finds
likely. For content the model finds unlikely, SFT is the only one of the three
that learned it here, and the hardest to dose.

Caveats: 24 training rows, one seed, greedy probes, keyword scoring.
