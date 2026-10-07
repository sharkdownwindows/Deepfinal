import json
import sys
from pathlib import Path
from types import ModuleType

import yaml
from PIL import Image

from personalized_t2i.inference.generate import (
    generate_evaluation_batch,
    validate_generation_completeness,
)


def test_missing_generation_output_is_rejected():
    import pytest

    with pytest.raises(ValueError, match="matrix incomplete"):
        validate_generation_completeness(
            [],
            prompts=[{"id": "p01"}],
            seeds=[11],
            modes=["base"],
            run_id="run",
        )


def test_generate_matrix(tmp_path, monkeypatch):
    class FakeTorch(ModuleType):
        float16 = "float16"
        float32 = "float32"

        class cuda:
            @staticmethod
            def is_available():
                return False

            @staticmethod
            def empty_cache():
                return None

    class FakeGenerator:
        def __init__(self, seed):
            self.seed = seed

    class FakePipeline:
        def __init__(self):
            self.loaded_adapter = False

        def load_lora_weights(self, *args, **kwargs):
            self.loaded_adapter = True

        def __call__(self, **kwargs):
            return type(
                "Result",
                (),
                {"images": [Image.new("RGB", (8, 8), (10, 20, 30))]},
            )()

    monkeypatch.setitem(sys.modules, "torch", FakeTorch("torch"))
    run_id = "r"
    artifact_dir = tmp_path / "artifacts" / run_id
    artifact_dir.mkdir(parents=True)
    pilot_metadata = artifact_dir / "pilot_metadata.jsonl"
    pilot_metadata.write_text('{"pilot": true}\n', encoding="utf-8")
    evaluation_metrics = tmp_path / "results" / "metrics_per_sample.csv"
    evaluation_metrics.parent.mkdir(parents=True)
    evaluation_metrics.write_text("completed-evaluation\n", encoding="utf-8")
    config_path = artifact_dir / "config.resolved.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "id": "stable-diffusion-v1-5/stable-diffusion-v1-5",
                    "revision": "immutable-revision",
                },
                "data": {"concept_id": "dog_plush"},
                "inference": {
                    "seeds": [11, 22, 33, 44],
                    "resolution": 512,
                    "num_inference_steps": 30,
                    "guidance_scale": 7.5,
                    "scheduler": "fixed",
                },
            }
        ),
        encoding="utf-8",
    )
    adapter_path = artifact_dir / "adapter"
    adapter_path.mkdir()
    created_generators = []
    pipeline = FakePipeline()

    def make_generator(seed):
        generator = FakeGenerator(seed)
        created_generators.append(generator)
        return generator

    metadata_path = generate_evaluation_batch(
        run_id=run_id,
        adapter_path=str(adapter_path),
        artifacts_root=str(tmp_path / "artifacts"),
        pipeline_factory=lambda *args: pipeline,
        generator_factory=make_generator,
    )

    with open(metadata_path, encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream]
    assert len(records) == 8 * 4 * 2
    assert {record["generation_mode"] for record in records} == {"base", "adapter"}
    assert len(created_generators) == len(records)
    assert len({id(generator) for generator in created_generators}) == len(records)
    assert all(Path(record["image_path"]).is_file() for record in records)
    assert pilot_metadata.read_text(encoding="utf-8") == '{"pilot": true}\n'
    assert evaluation_metrics.read_text(encoding="utf-8") == "completed-evaluation\n"

    import pytest
    completed_metadata = Path(metadata_path).read_text(encoding="utf-8")
    with pytest.raises(FileExistsError, match="Generation metadata already exists"):
        generate_evaluation_batch(
            run_id=run_id,
            adapter_path=str(adapter_path),
            artifacts_root=str(tmp_path / "artifacts"),
            pipeline_factory=lambda *args: pipeline,
            generator_factory=make_generator,
        )
    assert Path(metadata_path).read_text(encoding="utf-8") == completed_metadata
