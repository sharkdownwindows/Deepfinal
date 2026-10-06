# DATA-01 Verification

## Scope

Select and document three non-human concepts with a class noun, unique token,
and ownership/source record.

## Selected concepts

| Concept ID | Class noun | Unique token | Owner | Source / permission record |
|---|---|---|---|---|
| `cat_mug` | mug | `zzobj01` | `project_team` | `self-captured` |
| `dog_plush` | plush toy | `zzobj02` | `project_team` | `self-captured` |
| `blue_white_vase` | vase | `zzobj03` | `project_team` | `self-captured` |

The versioned records are in `data/manifests/concepts.csv`; per-concept image
manifests use the same concept IDs, captions, source, and permission fields.
These are ordinary non-human object categories selected for the coursework
experiment.

## Risk screen and handling rules

- Use only the generic, unbranded object concept. Exclude recognizable
  copyrighted characters, logos, artwork, or other third-party designs unless
  the project records permission or a license that permits the intended use.
- Do not include people, personal identifiers, private documents, or sensitive
  information in source or evaluation images.
- If an image contains an excluded element, do not add it to the dataset until
  the issue is removed or permission is documented.
- Keep source photographs private and ignored by Git; commit only safe metadata
  and hashes.

## Verification limits

The user-provided collection was visually reviewed as contact sheets after the
project manifests passed file/hash validation. It contains photos of the three
listed objects, with no people or personal documents visible and no obvious
brand logos. The mug and vase have decorative surface artwork; its creator or
separate license cannot be established from the photographs. Treat those
motifs as unverified third-party designs: keep the dataset and outputs private
to the course project, and confirm rights before any public release. The
collection is recorded as user-provided, self-captured project data; do not
claim independent rights to the printed designs.

## Result

The three concepts, concept-level exclusions, and available image-level risk
review are documented. Manifest source and permission declarations pass the
DATA-02 validator.
