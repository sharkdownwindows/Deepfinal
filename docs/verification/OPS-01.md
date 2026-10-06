# OPS-01 Verification

## Scope
Repository and environment contract for reproducible LoRA experiments.

## Verification

- Repository structure exists according to the project planning specification.
- `.gitignore` excludes secrets, credentials, model caches, checkpoints, generated outputs, and local environments.
- `requirements.in` exists as the dependency input.
- `configs/schema.yaml` defines the core experiment configuration contract.
- `configs/environment.schema.yaml` defines reproducibility environment metadata.
- `src/personalized_t2i/config.py` validates required configuration fields and core sweep constraints.
- Config tests: `3 passed`.
- `git diff --check`: passed; only Windows LF/CRLF warnings were reported.
- Commit: `05b08d5` (`ops: define config and environment contracts`).
- Working tree verified clean after the OPS-01 commit.

## Model revision note

`configs/base.yaml` records model ID `stable-diffusion-v1-5/stable-diffusion-v1-5` and immutable revision `451f4fe16113bff5a5d2269ed5ad43b0592e9a14`. The revision was checked against the [upstream model repository](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/451f4fe16113bff5a5d2269ed5ad43b0592e9a14). This records the selected source revision; downloading the model and running the CUDA/model pilot remain unverified.

## Result

OPS-01 repository and environment contract acceptance criteria are represented in the repository: structure, ignore rules, dependency input, model ID/revision, and environment metadata schema. Model download, CUDA pilot, and training remain unverified.
