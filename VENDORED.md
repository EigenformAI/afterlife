# Vendored code

`src/afterlife/trainer/` and `tests/trainer/` are copied from
[EigenformAI NSL2](https://github.com/EigenformAI) at commit `f00840f6b94edd2892862c9dd8351571131f3212`
(`src/train/` and `tests/test_train/`).

| Afterlife | NSL2 |
|---|---|
| `trainer/qlora.py` | `src/train/qlora.py` |
| `trainer/artifacts.py` | `src/train/artifacts.py` |
| `trainer/asft.py`, `trainer/asft_trainer.py` | `src/train/asft.py`, `src/train/asft_trainer.py` (themselves adapted from unslothai/unsloth PR #4239; see `ASFT_SOURCE_COMMIT`) |
| `trainer/dft_fused.py` | `src/train/dft_fused.py` |

Local changes are limited to:

- `src.train.*` imports → `afterlife.trainer.*` (code and test patch targets)
- argparse description string in `qlora.py`
- `_build_query_response_sft_row` (`qlora.py`): the loss now covers only the
  response content plus the end-of-turn token. NSL2's version also trained on
  the assistant turn marker, any template scaffold before the response (e.g.
  Qwen3's empty `<think></think>`), and the newline after `<|im_end|>`. On
  Qwen3-8B those cost ~25, ~21 and ~23 nats per row under the base model, and
  training on the empty think block taught adapters to stop thinking. The NSL2
  behaviour is kept as `_build_query_response_sft_row_nsl` and used as the
  fallback when the response text can't be found on a token boundary. NSL2 is
  probably affected too (at least the turn marker and trailing newline).

Keep the diff against NSL2 that small: Afterlife calls `train_sft()` from
`afterlife/worker.py`, so fixes can be ported in either direction by copying
files. If both repos keep changing the trainer, split it into a shared package.
