# Blind human evaluation rubric v1

**Status: proposed frozen rubric v1 for Evaluation Lead review.** Use with the 60-sample plan in
`configs/human_eval_v1.yaml`. The form hides configuration and run identity,
randomizes sample order in the browser, shows the generated output beside the
three held-out reference photos, and displays the prompt being evaluated.

## Sampling plan

Select exactly one fixed prompt and seed for each cell in
`3 concepts × 5 representative configurations × 4 prompt categories = 60`.
The preselected prompts are p01 (simple), p03 (new background), p05
(viewpoint/action), and p08 (challenging composition); style prompt p07 is
outside this 60-image subset. The current protocol names the four-category
stratification count but not the category names. This explicit mapping is an
operationalization for reviewer confirmation before ratings are collected; do
not distribute the packet until the Evaluation Lead approves it. Use
generation seed 11, the first seed in the
frozen DATA-04 seed list. This is a stratified qualitative subset, not a random
sample of all possible prompts. Never replace a missing output with another
seed or prompt after looking at results. Stop packet preparation and report the
missing cell.

Only real generated outputs are eligible. The QA CPU placeholder mode and
smoke artifacts are rejected. Keep the private run map and blind key out of the
rating packet and out of Git. Do not reveal configuration names until the
ratings have been collected and locked.

## Instructions to raters

Rate each generated image independently. Compare it with the three held-out
reference photos and the shown prompt. Judge visible evidence only. Do not
guess which training configuration produced an image. Do not enter your name,
email, or other identifying details. Use the anonymous rater code assigned by
the organizer. Save the downloaded CSV and return it to the organizer without
editing its IDs or score columns. A blank score is missing; it is never zero.

## Fixed 1–5 scales

For each dimension, choose one score. Scores 2 and 4 represent intermediate
cases between the anchored descriptions.

| Score | Subject fidelity | Prompt adherence | Visual quality / artifacts |
|---|---|---|---|
| 1 | Target subject is absent or clearly the wrong identity/category. | Main requested scene, action, or relation is absent or contradicted. | Severe defects make the image unusable or hard to interpret. |
| 2 | Weak resemblance; major identity features are wrong or missing. | Only a small part of the prompt is followed; major elements are wrong. | Major visible defects substantially distract from the subject. |
| 3 | Recognizable target, with noticeable identity drift or missing traits. | Main request is partly satisfied, with a noticeable omission or mismatch. | Usable overall, with clear defects that reduce quality. |
| 4 | Strong resemblance with minor identity differences. | Main request is followed with only a minor mismatch or omission. | Clean and coherent with minor visible defects. |
| 5 | Very strong match to the shared subject traits in the references. | All salient requested content and relations are clearly present. | Clean, coherent, and free of distracting defects. |

Do not penalize a deliberate style prompt merely for being non-photographic.
For a `new_background` or `challenging_composition` prompt, judge the requested
setting and object relationships under prompt adherence. Keep subject identity
separate from visual polish.

## Failure tags

Select every tag supported by the image; leave tags blank when none apply.

| Tag | Meaning |
|---|---|
| `identity_drift` | Subject-specific shape, colors, markings, or other stable traits differ from the references. |
| `background_leakage` | Unrequested training-scene/background details dominate or recur. |
| `pose_copy` | A training pose or crop appears copied when the prompt requests a different view/action. |
| `prompt_refusal` | The image ignores or replaces the requested setting, view, action, or relation. |
| `memorization` | Image appears to reproduce a training image nearly verbatim. |
| `structure_artifact` | Broken, merged, duplicated, or implausible subject parts. |
| `rendering_artifact` | Visible texture, geometry, text, or rendering defects not covered above. |
| `other` | A material failure not represented by the listed tags; explain briefly in the optional comment. |

## Rater and data handling

Recruit at least 5 independent raters; target 8. Prefer raters outside the
project team and record only anonymous rater codes. Each rater receives the
same 60 samples in an independently randomized order. Preserve every returned
source CSV unchanged as `results/human_ratings.csv` (or another immutable raw
path outside Git if the team elects not to commit ratings). Combine files by
appending rows into a new source copy; do not correct or average source rows in
place. Run aggregation into a separate derived file. Record the actual number
of unique raters and complete ratings; list missing ratings separately.

The HTML packet and private key are generated only after all 15 run outputs and
three held-out references per concept are present. This repository currently
contains no real generated images or held-out reference files, so no ratings
may be collected from its current artifacts. The existing CPU smoke outputs
are testing fixtures, not rating stimuli.
