# Colab training status — 2026-10-07

## Observation

- Checked at: `2026-10-07 04:21 +07:00` (Asia/Saigon).
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

## Not verified

The training cell has no saved output. No live step count, timestamped log line,
checkpoint directory, return code, or final `pytorch_lora_weights.safetensors`
is visible. The public viewer is not attached to the user's private runtime and
shows `Sign in`, `Changes will not be saved`, and `Connect T4`.

Consequently, this snapshot does **not** prove that training is running, failed,
or completed. It also does not satisfy ML-01, ML-02, QA-01, ML-04, or ML-05,
because the direct notebook path does not yet provide the repository's complete
artifact/status contract or a successful adapter reload/evaluation.

## Evidence required for the next update

Record a new status only when at least one of these becomes visible and can be
timestamped:

- current step and total steps from the training log;
- a complete `checkpoint-<step>` with optimizer and random-state files;
- a nonzero process return code and its error log;
- final adapter path plus successful load/inference evidence;
- session wall time, source commit, resolved-config hash, and environment file.

Do not infer progress from the presence of the notebook cells alone. Do not run
or restart the user's training session while monitoring.
