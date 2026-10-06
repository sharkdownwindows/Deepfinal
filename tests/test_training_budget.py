"""Keep CPU smoke settings out of the fixed-compute experiment configs."""
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def read_config(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_main_training_configs_use_protocol_budget():
    paths = [ROOT / "configs/base.yaml"]
    for directory in ("data_sweep", "ml05_sweep", "rank_sweep"):
        paths.extend((ROOT / "configs" / directory).glob("*.yaml"))
    for path in paths:
        config = read_config(path)
        # Draft files without training settings are not executable runs.
        if not config or "training" not in config:
            continue
        assert config["training"]["max_train_steps"] == 500, str(path)


def test_smoke_output_is_separate_from_experiment_outputs():
    smoke = read_config(ROOT / "configs/qa_smoke.yaml")
    assert smoke["training"]["max_train_steps"] == 1
    for path in (ROOT / "configs").rglob("*.yaml"):
        if path.name == "qa_smoke.yaml":
            continue
        config = read_config(path)
        if not isinstance(config, dict) or "training" not in config:
            continue
        if "run" in config and isinstance(config["run"], dict):
            assert smoke["run"]["id"] != config["run"].get("id"), str(path)
        if "output" in config and isinstance(config["output"], dict):
            assert smoke["output"]["output_dir"] != config["output"].get("output_dir"), str(path)
