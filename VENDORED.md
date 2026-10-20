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

Keep the diff against NSL2 that small: Afterlife calls `train_sft()` from
`afterlife/worker.py`, so fixes can be ported in either direction by copying
files. If both repos keep changing the trainer, split it into a shared package.
