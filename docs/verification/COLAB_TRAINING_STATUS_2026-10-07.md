# Colab training status — 2026-10-07

## Observation

- Checked at: `2026-10-07 05:53 +07:00` (Asia/Saigon).
- Notebook: https://colab.research.google.com/drive/1ZHxG2TEwiRN0gFB1xtlILvpF_Qy8xp8i?usp=sharing
- Notebook title: `Untitled0.ipynb`.
- Observation method: reload and read the saved public notebook state without
  running, stopping, or editing any cell.

## Verified saved state

The saved outputs show:

- manifests for `cat_mug`, `dog_plush`, and `blue_white_vase`;
- 10 raw training-pool images reported for each concept;
- GPU observation `Tesla T4`;
- Diffusers `0.40.0` and the trainer-ready message;
- selected run `dog_plush_n5_r16_ts42`;
- instance prompt `a photo of zzobj02 plush toy`;
- five selected training images;
- rank 16, alpha 16, training seed 42, 500 maximum steps, and checkpointing
  every 100 steps;
- output directory
  `/content/gdrive/MyDrive/personalized-t2i/direct_training/dog_plush_n5_r16_ts42`.

## Verified training failure

The saved output of the training cell now shows a failed attempt for
`dog_plush_n5_r16_ts42`:

- training started from step 0 and loaded the Stable Diffusion v1.5 model
  components;
- PEFT failed while adding the LoRA adapter because the environment contains
  `torchao 0.10.0`, while the installed PEFT version reports that it supports
  only `torchao` versions above `0.16.0`;
- the underlying exception is
  `ImportError: Found an incompatible version of torchao`;
- `accelerate` returned nonzero exit status `1`;
- the notebook raised `AssertionError: Training lỗi` and points to
  `/content/gdrive/MyDrive/personalized-t2i/direct_training/dog_plush_n5_r16_ts42/train-20261006T211123Z.log`.

The log filename timestamps the failed session at `2026-10-06 21:11:23Z`
(`2026-10-07 04:11:23 +07:00`). This is a dependency incompatibility before the
first training step, not evidence of model convergence or partial completion.

## Not verified

No completed training step, complete checkpoint, final
`pytorch_lora_weights.safetensors`, successful adapter reload, or inference
output is visible. The public viewer is not attached to the user's private
runtime; its `Sign in`, `Changes will not be saved`, and `Connect T4` state is
unrelated to the failed private-runtime execution saved in the notebook.

Consequently, this snapshot records a reproducible failure signal but does not
satisfy ML-01, ML-02, QA-01, ML-04, or ML-05. The environment dependency set
must be corrected and the run repeated before those issues can advance.

## Evidence required for the next update

Record a new status only when at least one of these becomes visible and can be
timestamped:

- current step and total steps from the training log;
- a complete `checkpoint-<step>` with optimizer and random-state files;
- a subsequent explicit return code or a new error log after the dependency fix;
- final adapter path plus successful load/inference evidence;
- session wall time, source commit, resolved-config hash, and environment file.

Do not infer progress from the presence of the notebook cells alone. Do not run
or restart the user's training session while monitoring.
