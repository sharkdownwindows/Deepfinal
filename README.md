# Few-Shot Personalized Text-to-Image Generation with LoRA

This university project studies subject personalization of Stable Diffusion
with DreamBooth-LoRA. The planned system is a reproducible, configuration-driven
CLI pipeline for training, batch generation, evaluation, and reporting, plus a
thin local Gradio explorer for inference and result inspection. The experiment
is designed to measure how training-set size and LoRA rank affect subject
fidelity, prompt alignment, consistency, failure modes, and resource use.

## Course-project constraints

- Duration: 14 days.
- Team: six students with explicit owners and reviewers.
- This is a controlled coursework experiment, not a production platform or a
  claim of state-of-the-art performance.
- P0 work is the reproducible experiment pipeline and evidence. Infrastructure
  and UI features that do not support the research questions are secondary.

## Research questions

1. Under a fixed training-update budget, how does the number of reference
   images affect subject fidelity, prompt alignment, and consistency?
2. With a fixed reference set, how does LoRA rank affect subject fidelity,
   prompt alignment, overfitting, and adapter size?

These are empirical questions. The project does not assume that more images or
a higher rank will improve every outcome.

## Scope and non-goals

P0 scope:

- Subject personalization for three non-human concepts.
- Stable Diffusion v1.5 with a pinned revision, subject to the pilot decision.
- DreamBooth-LoRA on UNet attention with the text encoder frozen.
- Versioned manifests, nested data subsets, fixed prompts, and fixed seeds.
- Planned DINOv2 subject similarity, CLIP prompt similarity, blind human
  ratings, efficiency records, and failure analysis.
- Offline CLI execution and a local Gradio inference/result explorer.

Explicit non-goals:

- Multiple backbones, full-model fine-tuning, text-encoder tuning, multi-subject
  generation, style personalization, LoRA fusion, or hyperparameter search.
- A new model, loss, metric, or benchmark.
- Training through the UI.
- React, FastAPI, a database, authentication, job queues, cloud deployment, or
  production availability and latency guarantees.
- Claims beyond the tested model, concepts, prompts, seeds, and settings.

## Core experimental design

| Sweep | Data size | LoRA rank | Unique configurations per concept |
|---|---:|---:|---:|
| Data-size sweep | 1, 3, 5, 10 | 16 | 4 |
| Rank sweep | 5 | 4, 16, 32 | 3 |

The `n=5, rank=16` condition is shared, giving six unique configurations per
concept and 18 core training runs across three concepts. The planned dataset has
10 training-pool images and three held-out references per concept, with nested
subsets `D1 ⊂ D3 ⊂ D5 ⊂ D10`. Evaluation uses eight prompt IDs per concept and
four fixed generation seeds per prompt and run. Other training and inference
variables remain fixed within each sweep.

## System and pipeline overview

```text
manifests + experiment configs
            |
            v
validate -> train LoRA -> run artifacts -> batch generation
                                           |
held-out references + prompts ------------> evaluation
                                           |
                                           v
                           per-sample metrics -> aggregates
                                           |
                                           v
                             report assets + Gradio explorer
```

Training, batch generation, and evaluation are offline CLI stages. The local
filesystem is the artifact store; there is no database or job queue. Gradio is
limited to loading existing adapters, optional inference, and inspecting
precomputed results.

## Repository structure

| Path | Responsibility |
|---|---|
| `configs/` | Source of truth for experiment parameters and sweep definitions |
| `data/` | Private images plus versioned, trackable manifests |
| `prompt_bank/` | Evaluation prompts and generation seeds |
| `src/personalized_t2i/` | Reusable data, training, inference, evaluation, registry, and reporting logic |
| `scripts/` | Thin CLI entry points that call reusable code in `src/` |
| `app/` | Thin Gradio result explorer |
| `tests/` | Fast config, manifest, registry, and pipeline contract tests |
| `artifacts/` | Ignored per-run adapters, checkpoints, logs, generations, and metadata |
| `results/` | Reviewed source tables, figures, and qualitative grids |
| `notebooks/` | Exploration only; never the sole implementation of a required stage |
| `docs/` | Protocol, workflow, decisions, runbook, rubric, and planning documents |

## Setup prerequisites

- Python 3.10 or 3.11; the exact version is pending the environment pilot.
- An NVIDIA CUDA GPU for the planned training and inference stages. VRAM,
  driver, CUDA, and GPU-hour requirements remain unverified.
- Authorized access to the selected base model and frozen metric encoders.
- Three legally usable non-human concept datasets with documented provenance.
- Sufficient local storage for model caches and ignored run artifacts.

`requirements.in` records the planned top-level packages, but versions are not
yet resolved. `requirements.lock` is intentionally absent until the Python/CUDA
pilot selects a compatible environment and deterministic locking process.

## Minimal commands

The entry-point files exist, but all commands below are **Planned** and currently
exit with an explicit “not implemented” message.

| Stage | Command | Status |
|---|---|---|
| Validate data | `python scripts/validate_data.py --manifest <path>` | Planned |
| Train one run | `python scripts/train_run.py --config <path>` | Planned |
| Generate evaluation set | `python scripts/generate_eval_set.py --run-id <id>` | Planned |
| Evaluate outputs | `python scripts/evaluate_outputs.py --run-id <id>` | Planned |
| Build report assets | `python scripts/build_report_assets.py` | Planned |
| Open result explorer | `python app/demo.py` | Planned |

## Data and artifact policy

- Never commit raw or processed private images, held-out reference images,
  access tokens, model caches, LoRA adapters, checkpoints, or large generated
  batches.
- Commit schemas, manifests without private content, configs, prompts, source
  code, tests, documentation, and reviewed small result tables or figures.
- Store paths relative to the repository root and preserve the documented run
  ID layout when sharing artifacts outside Git.
- Keep failed runs, missing outputs, outliers, and unfavorable evidence visible;
  do not overwrite completed runs or cherry-pick only successful samples.
- The CSV files currently under `results/` contain headers only and are not
  experimental results.

## Reproducibility contract

Every completed run must record its Git commit, resolved configuration, dataset
version and file hashes, base-model ID and revision, Python/package/CUDA/driver
and GPU information, training and generation seeds, prompt-bank version, status,
logs, adapter/checkpoints, and generated-sample metadata. A generated sample
must be traceable to its prompt, seed, run, config, and dataset version.

Reproducibility means rerunning the pipeline and recovering the reported trend
within a stated tolerance; bitwise-identical output across different hardware is
not required. Failed runs remain registered with their error information.

## Team workflow

The six primary roles are technical product management, ML training, data and
evaluation, backend/MLOps, frontend/visualization, and QA/research communication.
Every artifact has an owner and reviewer, and at least two members must be able
to run the main pipeline. The team uses a short daily sync, explicit handoffs,
milestone quality gates, and a maximum of two active issues per member.

RQ1/RQ2 and the 18 core runs take priority over P1 work. Protocol v1 is frozen;
changes require a new protocol version, a decision record, and consistent reruns
of affected experiment cells.

## Current implementation status

- Planning documents and the initial repository scaffold exist.
- Config, prompt-bank, source, test, CLI, and Gradio files are placeholders.
- Training, generation, automated evaluation, reporting, and the Gradio explorer
  are not implemented or verified.
- No model or dataset has been downloaded, and no GPU experiment has been run.
- No experimental metrics or findings are available.
- Dependency versions, the model revision, and dataset manifests remain pending.
- Protocol v1 was approved by the ML Lead and Evaluation Lead and frozen on
  2026-09-29.

## Documentation

- [Project brief](docs/planning/01_PROJECT_BRIEF.md)
- [Product requirements](docs/planning/02_PRD.md)
- [Technical specification](docs/planning/03_PROJECT_OVERVIEW_AND_TECHNICAL_SPEC.md)
- [Team workflow and roadmap](docs/planning/04_TEAM_WORKFLOW_AND_ROADMAP.md)
- [Project structure and contracts](docs/planning/05_PROJECT_STRUCTURE.md)
- [Issues and milestones](docs/planning/06_ISSUES_AND_MILESTONES.md)
- [Runbook](docs/runbook.md)
- [Decision log](docs/decision_log.md)
- [Evaluation rubric](docs/evaluation_rubric.md)

## License and attribution status

The repository does not currently declare a project license. Model, dataset, and
dependency attributions are also pending final environment and dataset choices.
Before any experiment, record the selected model revision and license, dataset
ownership or license, and required dependency notices in
[LICENSES.md](LICENSES.md). Do not redistribute model weights or private data
through this repository.
