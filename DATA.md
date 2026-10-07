# Dataset

## Dataset package

The project dataset is a team-collected, versioned image dataset for three
non-human concepts. The manifests identify the source as `self-captured` and
the consent/license field as `self-captured`. The private Drive archive contains
the canonical source images, held-out references, and manifests:

- [Download `personalized-t2i-dataset-v1.zip` from Google Drive](https://drive.google.com/file/d/1w76esm9xl3xnRe4MYB835FymYq4PH8U9/view?usp=drivesdk)
- Drive folder: [Personalized-T2I-Dataset-v1](https://drive.google.com/drive/folders/1oNCVUgvSuRW5y_0UjBEX-7shFQlpc3-J)
- Archive SHA-256: `925553f09781aa6e1eea27b7a9b8b3749ec13c159314f5bcdbfb656ea99ab776`

The archive is private to the connected Drive account. Share it only with the
project team and instructor after confirming they are authorized to access the
images. Do not make it public or commit the images to Git.

## Contents and split

Extract the archive at the repository root. It contains a `data/` directory:

```text
data/
├── raw/             # 30 training-pool source images (10 per concept)
├── eval_refs/       # 9 held-out reference images (3 per concept)
├── manifests/       # 3 concept manifests, concept registry, README
└── README.md
```

The concepts are `cat_mug` (`zzobj01`), `dog_plush` (`zzobj02`), and
`blue_white_vase` (`zzobj03`). Each concept has ten training-pool images and
three separate held-out images. The training pool defines nested subsets
`D1 ⊂ D3 ⊂ D5 ⊂ D10`; `subset_membership` in the versioned manifest identifies
which images belong to each subset. Held-out images are for evaluation only and
must never enter a training subset.

The three `*_v1.csv` manifests contain relative paths and SHA-256 checksums for
all 39 images. `concepts.csv` records the concept IDs, unique tokens, dataset
version, and provenance fields. There is no third-party dataset URL: these
images are recorded in the manifests as collected by the project team.

## Restore and verify

Download and extract the archive into the repository root, merging its `data/`
directory with the repository. From the repository root, validate each manifest
in the project environment:

```powershell
python scripts/validate_data.py --manifest data/manifests/cat_mug_v1.csv
python scripts/validate_data.py --manifest data/manifests/dog_plush_v1.csv
python scripts/validate_data.py --manifest data/manifests/blue_white_vase_v1.csv
```

The validator checks the manifest contract, expected 10/3 train/held-out split,
relative paths, image files, and SHA-256 values. Do not edit the images without
creating a new dataset version and updating the corresponding manifest hashes.

## Preprocessing

`data/processed/` is derived data and is intentionally excluded from the Drive
archive. It is not needed to restore or verify the dataset. The repository
implementation in `src/personalized_t2i/data/prepare.py` deterministically
assigns the locked nested subsets, applies EXIF orientation, converts to RGB,
rejects non-square source images, resizes to 512×512 with Lanczos, and saves
JPEG at quality 95 without random flips. Existing processed files were
previously checked against regenerated candidates for all 30 training images.

To rebuild the processed images from the canonical raw images, use the project
environment from the repository root (the package must be installed or `src`
must be on `PYTHONPATH`):

```powershell
python -m personalized_t2i.data.prepare
```

The resulting files and preprocessing provenance are written under
`data/processed/` and `artifacts/`. The training configs use the raw images and
perform their own configured training-resolution transforms; this 512×512
processed copy is useful for the locked preprocessing artifact and verification,
but does not need to be uploaded alongside the raw data.

## Data handling

Keep `raw/`, `eval_refs/`, and `processed/` out of Git. Keep held-out images
separate from training. Manifests and preprocessing code may be versioned in
the repository; image files and generated model artifacts belong in restricted
storage. Preserve the Drive archive as the shared source snapshot for this
dataset version.
