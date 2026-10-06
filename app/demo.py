"""Local Gradio result explorer for FE-01.

This interface only browses existing experiment artifacts and results.
Training is intentionally not performed through the UI.
"""

import json
import os
from pathlib import Path

import gradio as gr
import pandas as pd
import yaml


ARTIFACTS_DIR = Path("artifacts")
METRICS_PATH = Path("results/metrics_per_sample.csv")


# ---------------------------------------------------------------------
# Artifact discovery
# ---------------------------------------------------------------------

def discover_runs():
    """Discover runs that contain a status.json file."""
    if not ARTIFACTS_DIR.exists():
        return []

    runs = []

    for run_dir in sorted(ARTIFACTS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue

        status_path = run_dir / "status.json"

        if not status_path.exists():
            continue

        try:
            with status_path.open("r", encoding="utf-8") as f:
                status = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        config = load_config(run_dir)

        runs.append(
            {
                "run_id": run_dir.name,
                "status": status.get("status", "unknown"),
                "config": config,
            }
        )

    return runs


def load_config(run_dir):
    """Load resolved YAML configuration for a run."""
    config_path = run_dir / "config.resolved.yaml"

    if not config_path.exists():
        return {}

    try:
        with config_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return {}


def get_concept_id(run):
    """Return concept ID from the resolved config."""
    config = run.get("config", {})
    data = config.get("data", {})

    if data.get("concept_id"):
        return str(data["concept_id"])

    # Compatibility with older configs.
    token_to_concept = {
        "zzobj01": "cat_mug",
        "zzobj02": "dog_plush",
        "zzobj03": "blue_white_vase",
    }

    token = data.get("instance_token")

    if token in token_to_concept:
        return token_to_concept[token]

    return "unknown"


def get_run(run_id):
    """Return discovered run information."""
    for run in discover_runs():
        if run["run_id"] == run_id:
            return run

    return None


# ---------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------

def metadata_path_candidates(run_id):
    """Return supported metadata locations."""
    run_dir = ARTIFACTS_DIR / run_id

    return [
        run_dir / "generations" / "metadata.jsonl",
        run_dir / "generated" / "metadata.jsonl",
    ]


def load_metadata(run_id):
    """Load generation metadata for a run."""
    for metadata_path in metadata_path_candidates(run_id):
        if not metadata_path.exists():
            continue

        records = []

        try:
            with metadata_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()

                    if not line:
                        continue

                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        except OSError:
            return []

        return records

    return []


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def load_metrics():
    """Load per-sample evaluation metrics."""
    if not METRICS_PATH.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(METRICS_PATH)
    except (OSError, pd.errors.ParserError):
        return pd.DataFrame()


# ---------------------------------------------------------------------
# Dropdown data
# ---------------------------------------------------------------------

def get_concept_choices():
    """Return available concept IDs."""
    concepts = sorted(
        {
            get_concept_id(run)
            for run in discover_runs()
            if get_concept_id(run) != "unknown"
        }
    )

    return concepts


def get_runs_for_concept(concept_id):
    """Return run IDs belonging to a concept."""
    runs = discover_runs()

    if not concept_id or concept_id == "All":
        return [run["run_id"] for run in runs]

    return [
        run["run_id"]
        for run in runs
        if get_concept_id(run) == concept_id
    ]


def get_prompt_choices(run_id):
    """Return prompt IDs available for a run."""
    records = load_metadata(run_id)

    return sorted(
        {
            str(record.get("prompt_id"))
            for record in records
            if record.get("prompt_id") is not None
        }
    )


def get_seed_choices(run_id, prompt_id):
    """Return seeds available for a run/prompt."""
    records = load_metadata(run_id)

    seeds = {
        record.get("seed")
        for record in records
        if str(record.get("prompt_id")) == str(prompt_id)
    }

    return sorted(seed for seed in seeds if seed is not None)


# ---------------------------------------------------------------------
# Image lookup
# ---------------------------------------------------------------------

def normalize_path(path_value):
    """Normalize Windows paths stored in JSON metadata."""
    if not path_value:
        return None

    path_text = str(path_value).replace("\\", os.sep)
    path = Path(path_text)

    if path.exists():
        return path

    return None


def find_image(run_id, prompt_id, seed):
    """Find an image using artifact conventions and metadata."""
    run_dir = ARTIFACTS_DIR / run_id

    prompt_id = str(prompt_id)
    seed = str(seed)

    # Official artifact layout.
    candidates = [
        run_dir / "generations" / prompt_id / f"{seed}.png",
        run_dir / "generated" / prompt_id / f"{seed}.png",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Metadata fallback.
    records = load_metadata(run_id)

    for record in records:
        if (
            str(record.get("prompt_id")) == prompt_id
            and str(record.get("seed")) == seed
        ):
            image_path = normalize_path(record.get("image_path"))

            if image_path is not None:
                return image_path

    return None


# ---------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------

def provenance_text(run_id):
    """Build provenance information for a run."""
    run = get_run(run_id)

    if run is None:
        return "No run selected."

    config = run.get("config", {})
    data = config.get("data", {})
    training = config.get("training", {})
    model = config.get("model", {})

    status_path = ARTIFACTS_DIR / run_id / "status.json"

    status_data = {}

    if status_path.exists():
        try:
            with status_path.open("r", encoding="utf-8") as f:
                status_data = json.load(f)
        except (OSError, json.JSONDecodeError):
            pass

    return (
        f"Run ID: {run_id}\n"
        f"Status: {status_data.get('status', run.get('status', 'unknown'))}\n"
        f"Concept ID: {get_concept_id(run)}\n"
        f"Data size: {data.get('subset_size', 'unknown')}\n"
        f"LoRA rank: {training.get('rank', 'unknown')}\n"
        f"LoRA alpha: {training.get('alpha', 'unknown')}\n"
        f"Training seed: {training.get('seed', 'unknown')}\n"
        f"Base model: {model.get('id', 'unknown')}\n"
        f"Instance token: {data.get('instance_token', 'unknown')}\n"
        f"Instance prompt: {data.get('instance_prompt', 'unknown')}\n"
        f"Started at: {status_data.get('started_at', 'unknown')}\n"
        f"Completed at: {status_data.get('completed_at', 'unknown')}"
    )


def config_text(run_id):
    """Return the resolved configuration as readable YAML."""
    run = get_run(run_id)

    if run is None:
        return "No run selected."

    config = run.get("config", {})

    if not config:
        return "config.resolved.yaml not found or could not be read."

    return yaml.safe_dump(
        config,
        sort_keys=False,
        allow_unicode=True,
    )


def metrics_text(run_id, prompt_id, seed):
    """Return metrics for a selected sample."""
    metrics = load_metrics()

    if metrics.empty:
        return (
            "No evaluation metrics are available yet.\n"
            "This QA smoke artifact only contains generated placeholder images."
        )

    filtered = metrics[
        (metrics["run_id"].astype(str) == str(run_id))
        & (metrics["prompt_id"].astype(str) == str(prompt_id))
        & (metrics["generation_seed"].astype(str) == str(seed))
    ]

    if filtered.empty:
        return "No metrics found for this run / prompt / seed."

    row = filtered.iloc[0]

    lines = ["Evaluation Metrics"]

    for column in [
        "dino_subject_similarity",
        "clip_prompt_similarity",
        "lpips_diversity_optional",
        "valid",
        "invalid_reason",
    ]:
        if column in row.index:
            lines.append(f"{column}: {row[column]}")

    return "\n".join(lines)


def prompt_text(run_id, prompt_id):
    """Return the actual prompt text from metadata."""
    records = load_metadata(run_id)

    for record in records:
        if str(record.get("prompt_id")) == str(prompt_id):
            return record.get("prompt", "")

    return ""


# ---------------------------------------------------------------------
# UI callbacks
# ---------------------------------------------------------------------

def on_concept_change(concept_id):
    """Update run and comparison-run dropdowns."""
    runs = get_runs_for_concept(concept_id)

    selected = runs[0] if runs else None

    return (
        gr.Dropdown(
            choices=runs,
            value=selected,
            interactive=bool(runs),
        ),
        gr.Dropdown(
            choices=runs,
            value=selected,
            interactive=bool(runs),
        ),
    )


def on_run_change(run_id):
    """Update prompt, seed and comparison controls after run selection."""
    if not run_id:
        return (
            gr.Dropdown(choices=[], value=None, interactive=False),
            gr.Dropdown(choices=[], value=None, interactive=False),
            "No run selected.",
            "No run selected.",
            "No run selected.",
        )

    prompts = get_prompt_choices(run_id)
    prompt = prompts[0] if prompts else None

    seeds = get_seed_choices(run_id, prompt) if prompt else []
    seed = seeds[0] if seeds else None

    runs = discover_runs()

    run = get_run(run_id)
    concept = get_concept_id(run) if run else "unknown"

    comparison_runs = [
        item["run_id"]
        for item in runs
        if get_concept_id(item) == concept
    ]

    comparison_default = (
        comparison_runs[0] if comparison_runs else None
    )

    return (
        gr.Dropdown(
            choices=prompts,
            value=prompt,
            interactive=bool(prompts),
        ),
        gr.Dropdown(
            choices=seeds,
            value=seed,
            interactive=bool(seeds),
        ),
        provenance_text(run_id),
        config_text(run_id),
        gr.Dropdown(
            choices=comparison_runs,
            value=comparison_default,
            interactive=bool(comparison_runs),
        ),
    )


def on_prompt_change(run_id, prompt_id):
    """Update seeds when prompt changes."""
    if not run_id or not prompt_id:
        return gr.Dropdown(
            choices=[],
            value=None,
            interactive=False,
        )

    seeds = get_seed_choices(run_id, prompt_id)
    seed = seeds[0] if seeds else None

    return gr.Dropdown(
        choices=seeds,
        value=seed,
        interactive=bool(seeds),
    )


def load_result(run_id, prompt_id, seed):
    """Load one precomputed generation and its metrics."""
    if not run_id or not prompt_id or seed is None:
        return (
            None,
            "No result selected.",
            "No result selected.",
        )

    image_path = find_image(run_id, prompt_id, seed)

    if image_path is None:
        return (
            None,
            "No generated image found for this run/prompt/seed.",
            provenance_text(run_id),
        )

    metrics = metrics_text(run_id, prompt_id, seed)

    return (
        str(image_path),
        metrics,
        provenance_text(run_id),
    )


def load_comparison(
    run_id,
    comparison_run_id,
    prompt_id,
    seed,
):
    """Compare the same prompt/seed across two runs."""
    if not run_id or not comparison_run_id:
        return (
            None,
            None,
            "Select two runs to compare.",
        )

    if not prompt_id or seed is None:
        return (
            None,
            None,
            "Select a prompt and seed first.",
        )

    image_a = find_image(run_id, prompt_id, seed)
    image_b = find_image(comparison_run_id, prompt_id, seed)

    if image_a is None:
        image_a = None

    if image_b is None:
        image_b = None

    info = (
        "Same Prompt / Same Seed comparison\n\n"
        f"Prompt ID: {prompt_id}\n"
        f"Seed: {seed}\n\n"
        f"Run A: {run_id}\n"
        f"Run B: {comparison_run_id}\n\n"
        f"Prompt: {prompt_text(run_id, prompt_id)}\n"
    )

    return (
        str(image_a) if image_a else None,
        str(image_b) if image_b else None,
        info,
    )


# ---------------------------------------------------------------------
# Gradio application
# ---------------------------------------------------------------------

def build_app():
    """Build the Gradio application."""
    runs = discover_runs()
    concepts = get_concept_choices()

    initial_concept = concepts[0] if concepts else None

    run_choices = get_runs_for_concept(initial_concept)

    initial_run = run_choices[0] if run_choices else None

    prompts = get_prompt_choices(initial_run) if initial_run else []
    initial_prompt = prompts[0] if prompts else None

    seeds = (
        get_seed_choices(initial_run, initial_prompt)
        if initial_run and initial_prompt
        else []
    )

    initial_seed = seeds[0] if seeds else None

    with gr.Blocks(
        title="Personalized Text-to-Image Generation"
    ) as demo:

        gr.Markdown(
            "# Personalized Text-to-Image Generation\n"
            "## Local Result Explorer\n"
            "Browse existing experiment artifacts and evaluation results. "
            "Training is not performed through this interface."
        )

        if not runs:
            gr.Markdown(
                "### No experiment runs found\n"
                "No completed run artifacts are currently available "
                "under `artifacts/`."
            )

        with gr.Row():
            concept_dropdown = gr.Dropdown(
                label="Concept",
                choices=concepts,
                value=initial_concept,
                interactive=bool(concepts),
            )

            run_dropdown = gr.Dropdown(
                label="Run",
                choices=run_choices,
                value=initial_run,
                interactive=bool(run_choices),
            )

            comparison_run_dropdown = gr.Dropdown(
                label="Comparison Run",
                choices=run_choices,
                value=initial_run,
                interactive=bool(run_choices),
            )

        with gr.Row():
            prompt_dropdown = gr.Dropdown(
                label="Prompt",
                choices=prompts,
                value=initial_prompt,
                interactive=bool(prompts),
            )

            seed_dropdown = gr.Dropdown(
                label="Seed",
                choices=seeds,
                value=initial_seed,
                interactive=bool(seeds),
            )

        load_button = gr.Button(
            "Load Result",
            variant="primary",
        )

        with gr.Row():
            generated_image = gr.Image(
                label="Generated Image",
                type="filepath",
            )

            evaluation_metrics = gr.Textbox(
                label="Evaluation Metrics",
                lines=12,
                interactive=False,
            )

        provenance = gr.Textbox(
            label="Provenance",
            lines=14,
            interactive=False,
        )

        resolved_config = gr.Code(
            label="Resolved Configuration",
            language="yaml",
            interactive=False,
        )

        gr.Markdown("## Same Prompt / Same Seed")

        comparison_button = gr.Button(
            "Load Comparison View"
        )

        with gr.Row():
            comparison_image_a = gr.Image(
                label="Run A",
                type="filepath",
            )

            comparison_image_b = gr.Image(
                label="Run B",
                type="filepath",
            )

        comparison_info = gr.Textbox(
            label="Comparison Information",
            lines=8,
            interactive=False,
        )

        # -------------------------------------------------------------
        # Events
        # -------------------------------------------------------------

        concept_dropdown.change(
            fn=on_concept_change,
            inputs=concept_dropdown,
            outputs=[
                run_dropdown,
                comparison_run_dropdown,
            ],
        )

        run_dropdown.change(
            fn=on_run_change,
            inputs=run_dropdown,
            outputs=[
                prompt_dropdown,
                seed_dropdown,
                provenance,
                resolved_config,
                comparison_run_dropdown,
            ],
        )

        prompt_dropdown.change(
            fn=on_prompt_change,
            inputs=[
                run_dropdown,
                prompt_dropdown,
            ],
            outputs=seed_dropdown,
        )

        load_button.click(
            fn=load_result,
            inputs=[
                run_dropdown,
                prompt_dropdown,
                seed_dropdown,
            ],
            outputs=[
                generated_image,
                evaluation_metrics,
                provenance,
            ],
        )

        comparison_button.click(
            fn=load_comparison,
            inputs=[
                run_dropdown,
                comparison_run_dropdown,
                prompt_dropdown,
                seed_dropdown,
            ],
            outputs=[
                comparison_image_a,
                comparison_image_b,
                comparison_info,
            ],
        )

    return demo


def main():
    """Launch the local Gradio application."""
    demo = build_app()
    demo.launch()


if __name__ == "__main__":
    main()