# EVAL-05 — Failure taxonomy and case review

**Protocol:** failure taxonomy v1  
**Purpose:** inspect generated outputs, preserve representative failures, and separate base-model limitations from failures introduced by LoRA.

This document defines how reviewers classify evidence. The examples below are
hypothetical decision examples, not observations from this project. Do not cite
them as experimental results.

## Review unit and controls

Review an output together with its prompt, the three held-out subject references,
and the nearest relevant training image when copying is suspected. To attribute
a failure to the base model or LoRA, compare the base and adapted outputs using
the **same prompt, seed, dimensions, sampler, inference steps, guidance scale,
and model revision**. Save both image paths and the exact base-model revision.

Use one row per confirmed sample/tag in `results/failures.csv`. A sample can have
multiple tags, but each row must preserve `run_id`, `concept_id`, `prompt_id`,
and `generation_seed`. Keep a second reviewer’s disagreement in `notes`; use
`review_status=needs_review` until it is resolved. Never infer memorization
from high similarity alone without inspecting the nearest training image.

## Controlled taxonomy

| Tag | What to look for | Evidence to record |
|---|---|---|
| `underfit` | The output looks like a generic class instance and misses stable subject traits visible in held-out references. | Held-out reference comparison, DINO score when available, and rank/data-size context. |
| `identity_drift` | Stable subject traits change across prompts or seeds, or differ materially from held-out references. | Reference comparison and the exact prompt/seed; compare other outputs of the same run. |
| `background_leakage` | An unrequested training background or scene recurs in the output. | Prompt text, output, and the matching training image/background. |
| `pose_copy` | A training pose, viewpoint, or crop recurs despite a prompt requesting a different pose/view. | Prompt text, output, and matching training image; distinguish a repeated pose from a near-duplicate. |
| `prompt_refusal` | A requested setting, action, viewpoint, or relation is ignored or replaced. | Prompt text, output, CLIP/human evidence where available, and the paired base output. |
| `memorization` | The output reproduces a training image or a distinctive composition nearly verbatim. | Output and the specific nearest training image side by side; do not use held-out references for this comparison. |
| `structure_artifact` | Parts are broken, merged, duplicated, or physically implausible. | Output and a short description of the affected region. |
| `rendering_artifact` | Texture, edges, text, or rendering defects materially reduce readability or quality. | Output and a short description of the defect. |
| `other` | A material failure not covered above. | Describe the visible failure and why no existing tag fits. |

The failure tag describes the visible symptom. `attribution` is a separate
judgment with one of `base_model`, `lora`, `both`, or `uncertain`.

## Attribution rules

| Paired evidence (same prompt and seed) | Attribution |
|---|---|
| Base and LoRA outputs show the same scene/composition failure; no new failure is introduced by adaptation. | `base_model` |
| Base follows the prompt, while the LoRA output introduces the failure or loses subject/prompt performance. | `lora` |
| Both show a limitation and LoRA adds a separate or stronger failure. | `both` |
| Paired base output or required provenance is missing, or reviewers cannot distinguish the cause. | `uncertain` |

These labels describe evidence for the tested prompt and seed; they are not
claims about all prompts or the model family. Do not label an output
`base_model` or `lora` without the paired control. The validator enforces this
rule for attributed cases. An uncertain case is still useful evidence; explain
why attribution is unresolved.

### Hypothetical decision examples

- If both the base and LoRA omit the requested wooden desk in the same prompt,
  record the prompt failure and use `base_model`, unless LoRA adds a distinct
  failure.
- If the base output follows the requested scene but the LoRA output replaces
  it with the training background, record `background_leakage` and use `lora`.
- If the LoRA output resembles one training photograph, but the base comparison
  or nearest training image is unavailable, do not claim memorization or assign
  a cause; mark the case `uncertain` / `needs_review` pending evidence.

## Case selection and reporting

1. Use the frozen evaluation prompt/seed pairs and preserve all generated
   outputs, including failures and null results.
2. Inspect the same predefined set across configurations; do not select cases
   because their result supports a hypothesis.
3. Record all candidate failures, then select representative examples only
   after the full review. Keep one case per tag only if it is genuinely
   representative; report additional distinct cases when needed.
4. Include both successful and failed paired examples when discussing a tag.
   State the denominator reviewed for each run/configuration and list missing
   outputs separately.
5. Cite `case_id`, `run_id`, `prompt_id`, and `generation_seed` in notes, report,
   and figures. Report per concept/configuration before making any summary claim.
6. Treat metric values as supporting evidence, not ground truth. Do not treat
   generated images as independent subjects or use an uncalibrated metric
   threshold to declare a failure.

## Data contract and validation

`results/failures.csv` is an evidence register, not a synthetic example file.
Its columns are:

- Provenance: `case_id`, `run_id`, `concept_id`, `prompt_id`, `generation_seed`.
- Finding: `failure_tag`, `severity`, `notes`, `automated_evidence`.
- Attribution: `attribution`, `base_image_path`, `base_model_id`,
  `base_model_revision`, `lora_image_path`.
- Memorization evidence: `nearest_training_image` (required for the
  `memorization` tag).
- Review: `review_status`, `reviewer_id`.

Validate before freezing evidence:

```bash
python scripts/validate_failure_cases.py --input results/failures.csv
python scripts/validate_failure_cases.py --input results/failures.csv --check-files
```

The first command checks IDs, tags, attribution, and required evidence fields.
The second also checks that image paths exist relative to the repository root.
The validator is read-only; it never edits the source CSV.

## Current evidence status

The repository checkout used to prepare this workflow contains no real generated
evaluation images or held-out reference images. The CSV is therefore only a
header template, not evidence that failure cases were reviewed. Fill it only
from real ML-04/ML-05 outputs and paired base-model controls. EVAL-05 is not
complete until representative, traceable cases are recorded and reviewed.
