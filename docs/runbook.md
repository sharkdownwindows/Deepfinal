# Runbook

## SD 1.5 LoRA pilot

The one-step pilot config at `configs/pilot/dog_plush_n10_r16_ts42.yaml` is for
checking model load, LoRA training, adapter reload, one generated sample, elapsed
time, and observed GPU memory. It is not a core experiment result.

Install the project dependencies and clone the official Diffusers repository
into the runtime. Use a CUDA GPU runtime; the project wrapper refuses to run
training on CPU.

```bash
python -m pip install -e .
git clone https://github.com/huggingface/diffusers.git third_party/diffusers
python -m pip install -e third_party/diffusers
python -m pip install -r third_party/diffusers/examples/dreambooth/requirements.txt
python scripts/train_run.py --config configs/pilot/dog_plush_n10_r16_ts42.yaml --diffusers-script third_party/diffusers/examples/dreambooth/train_dreambooth_lora.py
```

On Colab, upload/copy the repository and the ten training images so the paths
in the config resolve from the repository root. Set the runtime accelerator to
GPU. If the Diffusers checkout is elsewhere, pass its script path with
`--diffusers-script` or set `DIFFUSERS_TRAIN_DREAMBOOTH_LORA`.

The wrapper reserves a unique run directory and saves the resolved config,
environment metadata, trainer log, checkpoints, adapter, one LoRA sample,
status, and measured timing/GPU memory under `artifacts/<run_id>/`. Completed
or failed run IDs are never overwritten. GPU memory samples use `nvidia-smi`
and include other processes sharing the GPU.

After training completes, generate the fixed base/adapter evaluation matrix
and score it against the held-out references:

```bash
python scripts/generate_eval_set.py \
  --run-id dog_plush_n5_r16_ts42 \
  --adapter-path artifacts/dog_plush_n5_r16_ts42/adapter
python scripts/evaluate_outputs.py \
  --run-id dog_plush_n5_r16_ts42 \
  --metric dino
```

Batch generation writes images under
`artifacts/<run_id>/generations/evaluation/{base,adapter}/` and records the
full prompt/seed/mode matrix in `metadata.jsonl`. Evaluation updates
`results/metrics_per_sample.csv`; it requires the held-out reference images
and pinned DINOv2 weights to be available in the runtime.

For the 500-step research runs, use the corresponding `configs/ml05_sweep/`
config only after the pilot passes and update each config to the finalized
dataset path and full model revision.

## QA vertical slice and ML-04 data sweep

Run the end-to-end pilot on a CUDA runtime. It performs one training step,
reloads the adapter, generates one pilot sample, then records DINOv2 and CLIP
scores:

```bash
python scripts/run_vertical_slice.py \
  --config configs/pilot/dog_plush_n10_r16_ts42.yaml \
  --diffusers-script third_party/diffusers/examples/dreambooth/train_dreambooth_lora.py
```

After the vertical slice passes, preflight the 12 ML-04 cells and then execute
them on a suitable Colab GPU. Each cell uses the fixed 500-step budget. Do not
start this sweep on the local GTX 1050 Ti with 4 GiB VRAM; use a GPU runtime
with enough free memory and preserve the generated status CSV and run artifacts.

```bash
python scripts/run_sweep.py --dry-run
python scripts/run_sweep.py \
  --diffusers-script third_party/diffusers/examples/dreambooth/train_dreambooth_lora.py
```

Each run's selected nested subset is hashed and copied under its ignored
`artifacts/<run_id>/training_data/`. The timestamped sweep report records
completed/failed cells and a separate owner-review decision for failures.

## Evidence status

- BE-01 config/run validation and ML-02 training wrapper are implemented.
- BE-02 registry/provenance, ML-03 batch generation, and EVAL-01 DINO scoring
  code paths are implemented and covered by unit/smoke tests.
- ML-01 GPU pilot execution is still pending. Do not report a successful
  train/inference run until the Colab/local artifact contains `status.json`
  with `completed` and `metrics.json` with timing and GPU-memory measurements.
- The default model revision and runtime versions must remain recorded with
  the run artifacts; the dependency input is not locked yet.
