# DATA-02 Verification

## Scope

Each concept manifest defines ten training-pool records and three held-out
records with path, SHA-256, split, caption where applicable, and source/permission
fields.

## Manifest metadata audit

| Concept | Train pool | Held out | SHA-256 format | Source/license and training captions | Held-out subset membership |
|---|---:|---:|---|---|---|
| `cat_mug` | 10 | 3 | Recomputed and matched on all 13 files | Present; validator PASS | Empty |
| `dog_plush` | 10 | 3 | Recomputed and matched on all 13 files | Present; validator PASS | Empty |
| `blue_white_vase` | 10 | 3 | Recomputed and matched on all 13 files | Present; validator PASS | Empty |

The manifest paths use `data/raw/<concept>/...` for training records and
`data/eval_refs/<concept>/...` for held-out records. Train subset membership is
recorded only on `train_pool` rows.

## Verification

Ran `scripts/validate_data.py` for all three manifests against the user-provided
dataset. Each passed schema, count, ID uniqueness, image readability, duplicate
bytes, recomputed SHA-256, split separation, source/license, caption, path
safety, and train/held-out overlap checks. The 30 training photos and 9 held-out
photos are stored under Git-ignored data directories; no image bytes were
added to Git.

## Result

All DATA-02 manifest and file-level validation criteria pass for the supplied
dataset.
