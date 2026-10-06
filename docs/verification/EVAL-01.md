# EVAL-01 Verification — DINOv2 Fidelity

## Implemented

- Builds a normalized DINOv2 centroid from the concept's three held-out
  references and verifies their names and SHA-256 hashes against the dataset
  manifest when scoring a run.
- Exports a finite subject-similarity score per sample and retains
  `concept_id`, prompt, seed, and generation mode in the per-sample metrics
  table before any aggregation.
- Base and adapter rows use different sample IDs, so one mode cannot overwrite
  the other's score.

## Verification evidence and limits

`tests/test_fidelity.py` includes a finite-score smoke test, metadata/error
handling, held-out manifest/hash checks, and metric upsert checks. The batch
test checks distinct base/adapter IDs. No pretrained DINOv2 weights were
downloaded and no real generated run was scored in this environment; the
reported scores must come from running evaluation against the Colab artifacts.

```bash
python scripts/evaluate_outputs.py \
  --run-id dog_plush_n5_r16_ts42 \
  --metric dino
```
