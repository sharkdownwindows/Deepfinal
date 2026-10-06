# EVAL-04 runbook

This implements the frozen 60-image blind-rating subset and a separate source-
preserving aggregator. It does not generate model outputs or recruit raters.

## Before packet preparation

ML-04/ML-05 must produce genuine generated images for the five representative
configurations on all three concepts. Keep the 15-row run map private; use one
row per concept/configuration pair:

```csv
concept_id,config_id,run_id
cat_mug,n1-r16,<actual-run-id>
cat_mug,n5-r16,<actual-run-id>
cat_mug,n10-r16,<actual-run-id>
cat_mug,n5-r4,<actual-run-id>
cat_mug,n5-r32,<actual-run-id>
```

Repeat the five rows for `dog_plush` and `blue_white_vase`. Run IDs must match
the directory names in `artifacts/<run-id>/generated/metadata.jsonl`. Ensure
`data/eval_refs/<concept_id>/` contains exactly the three held-out reference
images for each concept. Do not commit those images, the run map, or the
generated packet/key.

Then run:

```bash
python scripts/prepare_human_eval.py prepare \
  --run-map artifacts/private/human_eval_run_map.csv \
  --output-root artifacts/human_eval_v1
```

The command fails closed if a required sample/reference is missing, duplicated,
unreadable, or marked `cpu_smoke_placeholder`. It creates
`artifacts/human_eval_v1/human_eval_v1_blind_packet.zip` for raters and a
separate `private/blind_key.csv` that must remain with the study organizer.
The packet contains opaque sample IDs, prompts, sanitized image copies, and
randomized browser order. It contains no run/configuration IDs.

Distribute the same packet to at least 5 independent raters (target 8). Assign
each person a distinct anonymous code such as `R01`; do not collect their names
or emails in the rating file. Each rater opens `index.html`, scores all 60
images using the frozen rubric, and downloads a CSV. Keep each returned CSV as
received; combine them into a new raw source CSV by appending rows, without
editing IDs or scores. Keep the raw file outside Git if it contains private
study information.

## Aggregate after collection

After at least five distinct anonymous raters have each completed the 60-sample
packet, combine their CSV rows into `results/human_ratings.csv` (the checked-in
file is currently an empty schema only). Then run:

```bash
python scripts/prepare_human_eval.py aggregate \
  --ratings results/human_ratings.csv \
  --blind-key artifacts/human_eval_v1/private/blind_key.csv \
  --output results/human_ratings_aggregate.csv
```

Aggregation refuses duplicate rater/sample pairs, unknown blinded IDs, invalid
1–5 scores, fewer than five raters, fewer than five complete scores on any
sample, or overwriting an existing derived file. For a progress check only,
add `--allow-incomplete`; the report then exposes per-sample missing ratings.
It leaves the source CSV unchanged and writes per-sample means, standard
deviations, complete/missing rating counts, and failure-tag counts. Preserve
the source file and report actual unique-rater/sample counts with the results.

The currently committed `artifacts/` and `data/eval_refs/` directories contain
only `.gitkeep`; the existing generation code labels its outputs as CPU smoke
placeholders. Therefore packet generation and real participant ratings must
wait for genuine ML-04/ML-05 outputs and the held-out reference images.
