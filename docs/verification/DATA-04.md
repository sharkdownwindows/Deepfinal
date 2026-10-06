# DATA-04 Verification

## Scope

Freeze a versioned evaluation prompt bank and four fixed generation seeds.

## Frozen inputs

- Prompt bank: `prompt_bank/evaluation_prompts.yaml`, version `v1`.
- Seeds: `prompt_bank/generation_seeds.yaml`, version `v1`, values
  `[11, 22, 33, 44]`.
- Each of `cat_mug`, `dog_plush`, and `blue_white_vase` has eight distinct IDs
  (`p01`–`p08`) across the simple, new-background, viewpoint/action, style, and
  challenging-composition categories.
- Run configuration validation pins `prompt_bank_version: v1`, `prompt_count:
  8`, and the same ordered seed list.
- `src/personalized_t2i/inference/generate.py` loads this shared bank and seed
  list and records the prompt-bank version, prompt ID, prompt, and seed in each
  sample's metadata. It rejects a changed bank version, wrong prompt count,
  missing categories, or a changed seed list.
- Runtime loader check returned eight prompts for each concept and the exact
  four configured seeds.

## Verification limit

The current batch-generation function emits CPU smoke-test placeholders; it does
not yet run Stable Diffusion inference with a base model or LoRA adapter.
Therefore the frozen inputs and metadata path are verified, but real base-versus-
LoRA generations using this bank remain unverified.

## Result

The prompt and seed protocol is versioned and enforced at configuration and
batch metadata boundaries. Actual model outputs remain pending the inference
implementation and model/data availability.
