# Experiment Protocol

## Protocol metadata

| Field | Value |
|---|---|
| Version | v1 |
| Date | 2026-09-29 |
| Owner role | Technical PM/BA |
| Reviewer roles | ML Lead; Evaluation Lead |
| Status | Frozen |
| Approval date | 2026-09-29 |
| Source design | [Project overview and technical specification](planning/03_PROJECT_OVERVIEW_AND_TECHNICAL_SPEC.md) |

## Research questions and hypotheses

- **RQ1:** With a fixed training-update budget, how does the number of reference images affect subject fidelity, prompt adherence, and consistency?
- **RQ2:** With the reference set fixed, how does LoRA rank affect subject fidelity, prompt adherence, overfitting, and adapter size?
- **H1:** Increasing the reference set from 1 to 3–5 images improves fidelity more than increasing it from 5 to 10 images.
- **H2:** Rank 4 may underfit; rank 32 may improve fidelity while increasing overfitting or reducing prompt adherence.
- **H3:** Good consistency must accompany sufficiently high fidelity; high output-to-output similarity caused by mode collapse is not a good result.

These hypotheses are exploratory expectations and may be rejected.

## Scope and non-goals

The core study personalizes one Stable Diffusion v1.5 backbone with DreamBooth-LoRA on UNet attention for three non-human subject concepts. It varies reference-set size and LoRA rank, then compares generated outputs using the controls and measures below. The base model ID is `stable-diffusion-v1-5/stable-diffusion-v1-5`; its exact revision SHA must be recorded when obtained. The technical specification's SDXL pilot gate applies before freeze; backbone must be fixed across every run and cannot be selected after inspecting core results.

Non-goals are multiple backbones, full fine-tuning, text-encoder tuning, style personalization, multi-subject generation, LoRA fusion, hyperparameter search, new losses/metrics/benchmarks, and production API or cloud deployment. P1 investigations (including rank 8, inference-scale sweeps, extra training seeds, and alternate subset selection) are outside the core matrix.

## Core experiment matrix

| Cell | Training images (n) | LoRA rank (r) | Sweep |
|---|---:|---:|---|
| n1-r16 | 1 | 16 | Data size |
| n3-r16 | 3 | 16 | Data size |
| n5-r16 | 5 | 16 | Shared anchor |
| n10-r16 | 10 | 16 | Data size |
| n5-r4 | 5 | 4 | Rank |
| n5-r32 | 5 | 32 | Rank |

`n5-r16` is shared by the data-size and rank sweeps, so there are **6 unique configurations per concept** and **18 main runs across 3 concepts**. The contingency floor is **2 concepts × the same 6 configurations = 12 comparable runs**; do not reduce or vary the conditions by concept.

### Independent variables

- Reference subset size: n = 1, 3, 5, or 10, with nested subsets `D1 ⊂ D3 ⊂ D5 ⊂ D10`.
- LoRA rank: r = 4, 16, or 32; `lora_alpha = rank` (`alpha/r = 1`).

### Controlled variables

Use the same three concepts and preselected nested subsets; backbone/revision, DreamBooth-LoRA method, UNet targets (`to_k`, `to_q`, `to_v`, `to_out.0`), frozen text encoder, no prior preservation, preprocessing and captions, and all baseline training settings across cells. Baseline settings are 512 resolution, batch size 1, gradient accumulation 1, learning rate 1e-4, constant scheduler, zero warmup, 500 maximum training steps, checkpoint interval 100, fp16, no text-encoder training, zero LoRA dropout, center crop, no random flip, and training seed 42. Any allowed memory options (gradient checkpointing, xFormers, or 8-bit Adam) must be identical across all main runs and recorded.

All runs use the same **500-step training budget**. This is a fixed-compute interpretation: smaller subsets repeat their images more often. Therefore, RQ1 concerns reference-image count under a fixed update budget, not a pure causal effect of information quantity.

For inference, hold fixed the prompt bank (8 prompt IDs per concept), generation seeds (11, 22, 33, 44), 512×512 dimensions, 30 inference steps, guidance scale 7.5, one fixed scheduler, null negative prompt, LoRA scale 1.0, and a fresh generator per sample. Use the same prompts and seeds for base and adapted outputs. Record hardware and dependency versions; same seeds need not yield bit-identical outputs across hardware or library versions.

## Evaluation and reporting

- **Subject fidelity:** cosine similarity between frozen `facebook/dinov2-base` embeddings of generated images and the centroid of three held-out references per concept. Report per concept first; do not use training images as the primary reference. Audit a small subset with subject crops if feasible. DINO is a proxy metric, not ground truth; full-image scores may reflect background or layout.
- **Prompt adherence:** image-text cosine similarity from frozen `openai/clip-vit-base-patch32`, replacing the unique token with its class noun before text encoding. CLIP is a proxy metric, not ground truth; use human ratings to inspect fine-grained relations and actions.
- **Consistency:** report mean and standard deviation of subject fidelity across prompts and seeds. Low variation is favorable only with adequate mean fidelity. Do not use output-to-output similarity alone.
- **Overfitting and failure analysis:** nearest-reference perceptual similarity and manual tags for copied pose/background/crop, prompt refusal, identity drift, artifacts, and other failure cases. Intermediate checkpoints are for failure analysis or a selection rule fixed in advance.
- **Human ratings:** blind, randomized configuration order; show held-out references beside outputs; rate subject fidelity, prompt adherence, and visual quality/artifacts on a 1–5 scale. Target at least 5 independent raters (target 8). Rate the five representative configurations `n1-r16`, `n5-r16`, `n10-r16`, `n5-r4`, and `n5-r32` on the specified stratified sample of 60 outputs.
- **Efficiency:** adapter file size, trainable parameter count, training wall time, and peak VRAM when reliable.
- **Optional diversity:** mean pairwise LPIPS across seeds within a prompt, interpreted alongside fidelity; diversity from identity drift is not an improvement.

Report per-concept curves, descriptive effect sizes, missing samples, uncertainty/distributions where available, and qualitative examples. Do not treat generated images as independent subjects for statistical significance, combine metrics into one weighted score, use FID/Inception Score, or set absolute DINO/CLIP thresholds before baseline calibration.

## Claim boundary

Results apply only to the tested backbone and revision, three concepts, selected subsets, prompts, seeds, and settings. This is a descriptive/exploratory study over three concepts; it does not establish universal findings or causal conclusions. Findings must state limitations and include negative or null results. DINO and CLIP are proxies rather than ground truth.

## Fallback and change control

For compute shortage, preserve comparability: the contingency floor is 2 concepts × all 6 identical configurations (12 runs). If GPU outage or budget pressure prevents the target, prioritize comparable core cells, omit P1 work, retain failed-run evidence, and report the shortfall; do not silently substitute settings or claim incomplete cells as complete. For pilot OOM, the technical specification permits SD v1.5, fp16, gradient checkpointing, xFormers, and optional 8-bit Adam, provided any selected options are held constant across main runs. If metrics and human ratings disagree, report the disagreement and multi-view evidence rather than privileging one score.

After freeze, a protocol change requires a new protocol version, a decision-log entry, identification of affected run IDs/cells, and consistent reruns of all affected cells. Preserve failed runs and their errors; never overwrite completed artifacts.
