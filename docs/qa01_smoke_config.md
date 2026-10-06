# QA-01 smoke configuration

The main experiment budget is 500 training steps (see `docs/protocol.md`).
`configs/base.yaml` and the committed ML-05 rank-32 configuration use that budget.
The one-step CPU smoke configuration is `configs/qa_smoke.yaml`, with its own
run ID and output directory (`artifacts/qa01_cpu_smoke`).

From the repository root, in the installed project environment, use:

```bash
CUDA_VISIBLE_DEVICES="" python scripts/train_run.py --config configs/qa_smoke.yaml
```

For PowerShell, set `$env:CUDA_VISIBLE_DEVICES = ""` before running the Python
command. CPU selection matters: the current trainer selects its lightweight
smoke path only when CUDA is unavailable and `max_train_steps <= 1`.
The configured image directory must exist and contain images.

Smoke output is a pipeline fixture, not a trained LoRA adapter. Do not use it
for ML-04/ML-05 results or official evaluation. To repeat the smoke check, choose
a fresh run ID and matching output directory in a copy of this config.

Verify the configuration regression with:

```bash
python -m pytest tests/test_vertical_slice.py tests/test_training_budget.py -q
```

This fix resolves the configuration regression only. The existing generator
in `src/personalized_t2i/inference/generate.py` currently produces CPU placeholder
images. QA-01's real train/load/generate/score acceptance still requires a real
inference implementation and GPU-run evidence. Existing run artifacts are not
updated by changing these configs.
