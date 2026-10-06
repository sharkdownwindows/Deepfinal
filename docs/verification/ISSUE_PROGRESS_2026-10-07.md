# Issue progress audit — 2026-10-07

Snapshot: 04:21 Asia/Saigon. GitHub main and local HEAD: `5da945cd04`.
Local working tree is concurrently being edited by Codex Desktop for #13–#15;
this audit does not change implementation files or certify those pending edits.

## Sources and verification

- GitHub API/CLI: all 24 issues inspected; #1 closed, #2–#24 open.
- Planning documents 01–06, implementation, tests, and existing verification notes inspected.
- Shared Colab notebook: https://colab.research.google.com/drive/1ZHxG2TEwiRN0gFB1xtlILvpF_Qy8xp8i?usp=sharing
- Cheap validation: `python -m pytest tests/test_failures.py tests/test_aggregate.py -q -p no:cacheprovider --basetemp artifacts/audit_issues_19_24_tmp` → **14 passed**.
- A full-suite attempt aborted during test collection inside the local Anaconda
  NumPy/MKL BLAS initialization while importing Torch. This is an environment
  failure, not evidence that the full suite passed or that a test assertion failed.
- The saved notebook is visible anonymously, with `Sign in` and `Connect T4`.
  These describe this browser session, not the user's active runtime. Saved outputs
  are historical evidence; they cannot establish current training liveness.

## Progress by issue

| Issue | Observed evidence | Remaining acceptance evidence |
|---|---|---|
| #1 | Closed; frozen protocol documented | No new completion audit performed |
| #2 | Structure, ignore rules, schemas, full model revision, verification note | GitHub acceptance checklist remains open |
| #3 | Three concepts, tokens, class nouns and source declarations recorded | Refer to DATA-01 note for limits of ownership review |
| #4 | Saved Colab validation PASS: 10 train + 3 held-out per concept; hashes and split checks | No live dataset revalidation in this audit |
| #5 | Saved preprocessing PASS: nested subsets and 30 processed 512×512 RGB images | No live file revalidation in this audit |
| #6 | 8 prompts/concept, seeds 11/22/33/44; version v1 | Real base/adapter evaluation batch evidence |
| #7 | Saved SD1.5 base inference output: 20 steps; Tesla T4 environment | LoRA train/reload, wall time, peak VRAM and pilot artifacts |
| #8 | Validation implementation and cheap tests | Keep implementation evidence distinct from GPU evidence |
| #9 | Official Diffusers wrapper implementation | Successful real training and saved-adapter reload |
| #10 | Lifecycle/provenance implementation; registry tests pass | Real run artifacts still needed for experiment claims |
| #11 | Batch generation implementation; fake-pipeline tests documented | Real 8×4 base and adapter batch |
| #12 | DINO held-out-centroid implementation and tests documented | Real metric execution; optional crop audit if feasible |
| #13 | CLIP implementation exists; Desktop assigned | Desktop validation and real metric execution |
| #14 | Desktop adding vertical slice and forecast metadata | Real train→load→generate→score plus runtime/VRAM forecast |
| #15 | Desktop correcting subset handling and 12-run data sweep | 12 real comparable data-size runs and failure records |
| #16 | Existing rank script is incomplete/incompatible (below) | Three concepts × ranks 4/16/32 at n=5; reuse n5/r16; time/size evidence |
| #17 | Aggregate implementation exists; integration bugs reproduced (below) | Correct real per-sample/aggregate CSV with completeness accounting |
| #18 | Blind-packet/rating aggregation tooling and tests exist | Real 60-sample packet and ≥5 raters; target 8 |
| #19 | Failure taxonomy and validator exist; failures CSV has no cases | Traceable representative real cases and baseline comparison |
| #20 | Gradio explorer implementation merged | Verification on real artifacts and correct paired comparison |
| #21 | Chart/grid library implementations merged | CLI integration, real figures, schema compatibility and provenance |
| #22 | Report outline exists | Evidence-backed findings and limitations |
| #23 | No independent successful real reproduction evidence found | Clean setup, representative run, differences/nondeterminism |
| #24 | No completed presentation/offline demo evidence found | Slides, live/offline backup and rehearsal |

No meaningful overall completion percentage can be inferred from issue closure
count or code presence. M3's 18 completed runs are **unverified**, not proven absent.

## Concrete implementation gaps

1. `scripts/run_ml05_sweep.py` hardcodes only `dog_plush`, revision `451f4fe`,
   `v1_n5` paths and a JSON manifest incompatible with current CSV manifests.
   It measures the old `checkpoint/pytorch_lora_weights.safetensors` path rather
   than the current adapter contract. Skipped runs get `time_seconds=0`, which
   is not their observed training time. Corrupt status errors are swallowed.
2. `evaluation/aggregate.py` groups only by run ID, defaults to 32 samples,
   and rejects a realistic fixture of 32 base + 32 adapter rows:
   `dog_plush_n5_r16_ts42 has 64 rows, expected at most 32`.
   Raising the expected count to 64 would mix baseline and LoRA scores.
3. A synthetic row with `valid=False` and DINO score 0.8 produces `dino_mean=0.8`.
   Metric-specific validity needs an explicit contract: one metric failure must
   not inadvertently remove another valid metric, but invalid metric values
   must not silently contribute. Current tests explicitly permit some partial
   scores; preserve useful partial scoring through explicit fields/policy.
4. `scripts/build_report_assets.py` exits with “not implemented yet”. The
   tracked aggregate CSV header differs from current writer/chart expectations.
5. README's implementation status still describes all code as placeholders and
   manifests/model revision as pending; several verification notes are stale.

## Saved Colab progress and blockers

The current shared notebook is a direct official-Diffusers training path rather
than the repository runner. Its saved outputs establish:

- manifests for all three concepts and 10 raw training images per concept;
- a Tesla T4 observation, Diffusers 0.40.0, and a pinned trainer revision;
- selected run `dog_plush_n5_r16_ts42`, five manifest-selected images, rank 16,
  500 steps, checkpoint interval 100, and a Drive output directory;
- a resolved config and `pip freeze` step were executed in the saved snapshot.

After a reload at 04:21 Asia/Saigon, the training cell still had no saved output,
no step counter/log line, no checkpoint, and no final adapter evidence. The
viewer is anonymous and shows `Sign in`, `Changes will not be saved`, and
`Connect T4`; these describe the audit browser, not the user's private runtime.
Therefore current liveness, progress percentage, completion, and failure status
remain unverified. See `docs/verification/COLAB_TRAINING_STATUS_2026-10-07.md`.

Do not restart the user's active training or pull code into a running process.
For monitoring, distinguish saved notebook observations from timestamped live
logs/status artifacts. Save run ID, source commit, resolved config/hash, step,
total steps, observed time/VRAM and artifact location when those are accessible.

## Recommended handoff

Keep #13–#15 with Desktop. CLI can repair #17 and integrate #21 in a separate
checkout immediately. Repair #16 only after Desktop's shared sweep/runner
interface stabilizes; otherwise limit the work to an independent plan and tests.
Do not close experiment issues based on synthetic tests or scaffold code.
