# DATA-03 Verification

## Scope

Lock nested subsets and a consistent image preprocessing path.

## Evidence

- `docs/data_03_preprocessing.md` records the per-concept selection order and
  viewpoint rationale before results.
- `src/personalized_t2i/data/prepare.py` assigns subset membership from the
  locked prefix order and verifies `D1 ⊂ D3 ⊂ D5 ⊂ D10`; held-out records must
  have empty subset membership.
- The transform applies EXIF orientation, converts to RGB, rejects non-square
  sources, resizes to 512×512 with Lanczos, and saves JPEG at quality 95. The
  registered raw images are specified as square, so no geometric crop is
  needed before resize. No random flip is applied.
- Synthetic-fixture validation: `tests/test_manifest.py` and
  `tests/test_prepare.py` passed (18 tests).
- Ran the preprocessing pipeline on the supplied dataset. It verified all 30
  raw hashes, confirmed the existing processed files match the deterministic
  candidate byte-for-byte, and reported `processed total: 30/30` and
  `512x512 RGB: PASS`.

## Verification note

The processed dataset already supplied by the user passed the pipeline's
rebuild comparison; it was left intact. Provenance records were written under
ignored `artifacts/data03_validation/`.

## Result

Locked subset logic and preprocessing behavior pass both synthetic-fixture
tests and validation against the actual supplied dataset.
