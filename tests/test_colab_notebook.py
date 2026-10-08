"""CPU-only checks for the published Colab cells; no Drive or ML imports."""

import ast
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs/COLAB_TRAIN_DEEPFINAL.md"


def code_cells():
    return re.findall(r"```python\n([\s\S]*?)```", GUIDE.read_text(encoding="utf-8"))


def test_notebook_matches_guide_and_has_no_saved_outputs():
    notebook = json.loads((ROOT / "notebooks/colab_train_deepfinal.ipynb").read_text(encoding="utf-8"))
    cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(cells) == len(code_cells()) == 9
    for cell, source in zip(cells, code_cells()):
        assert "".join(cell["source"]).strip() == source.strip()
        assert cell["execution_count"] is None and cell["outputs"] == []
        compile(source, "colab-cell", "exec")


def test_accelerate_loader_receives_explicit_config_path(tmp_path):
    config_path = str(tmp_path / "accelerate.yaml")
    calls = []

    def write_basic_config(mixed_precision="no", save_location=config_path):
        calls.append((mixed_precision, save_location))

    # Accelerate 1.15.0 requires config_file; it has no argument default.
    def load_config_from_file(config_file):
        assert config_file == config_path
        assert calls == [("fp16", config_file)]
        return SimpleNamespace(num_processes=1, mixed_precision="fp16", distributed_type="NO")

    source = code_cells()[3]
    setup = source[source.index("write_basic_config(mixed_precision="):]
    setup = setup.split("from personalized_t2i.training.runner", 1)[0]
    exec(compile(ast.parse(setup), "accelerate-setup", "exec"), {
        "write_basic_config": write_basic_config,
        "load_config_from_file": load_config_from_file,
        "default_config_file": config_path,
    })


def persist_configs(root, configs):
    source = code_cells()[4]
    block = source[source.index("CONFIG_DIR ="):]
    exec(compile(block, "persist-configs", "exec"), {
        "CORE_ROOT": root, "SESSION": "test-session", "CONFIGS": configs,
        "ANCHOR": "test", "yaml": yaml,
    })


def test_config_cell_can_repeat_without_rewriting(tmp_path):
    configs = {"test": {"training": {"rank": 16}}}
    persist_configs(tmp_path, configs)
    path = tmp_path / "configs/test-session/test.yaml"
    before = (path.read_bytes(), path.stat().st_mtime_ns)
    persist_configs(tmp_path, configs)
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before


def test_config_cell_preserves_mismatched_config(tmp_path):
    persist_configs(tmp_path, {"test": {"training": {"rank": 16}}})
    path = tmp_path / "configs/test-session/test.yaml"
    before = path.read_bytes()
    with pytest.raises((AssertionError, ValueError), match="Config"):
        persist_configs(tmp_path, {"test": {"training": {"rank": 4}}})
    assert path.read_bytes() == before


def test_config_cell_can_finish_partial_session(tmp_path):
    configs = {"first": {"rank": 16}, "second": {"rank": 4}}
    persist_configs(tmp_path, {"first": configs["first"]})
    persist_configs(tmp_path, configs)
    path = tmp_path / "configs/test-session/second.yaml"
    assert yaml.safe_load(path.read_text(encoding="utf-8")) == configs["second"]



def run_archive_cell(core_root, anchor="dog_plush_n5_r16_ts42"):
    exec(compile(code_cells()[6], "archive-failed-run", "exec"), {
        "LEGACY_CORE_ROOT": core_root,
        "ANCHOR": anchor,
        "SOURCE_COMMIT": "9efc6408bb262a469355b62d6e92ed3a6bad9c32",
        "datetime": datetime,
        "timezone": timezone,
        "json": json,
    })


def make_failed_anchor(core_root, error="Diffusers trainer exited with code 1"):
    run_dir = core_root / "artifacts/dog_plush_n5_r16_ts42"
    (run_dir / "trainer_output").mkdir(parents=True)
    (run_dir / "status.json").write_text(json.dumps({
        "run_id": "dog_plush_n5_r16_ts42",
        "status": "failed",
        "error": error,
    }), encoding="utf-8")
    return run_dir


def test_archive_cell_moves_only_expected_failed_attempt_and_records_it(tmp_path):
    core_root = tmp_path / "core_v1"
    failed = make_failed_anchor(core_root)

    run_archive_cell(core_root)

    assert not failed.exists()
    archives = list((core_root / "failed_attempts").iterdir())
    assert len(archives) == 1
    record = json.loads((archives[0] / "archive_record.json").read_text(encoding="utf-8"))
    assert record["run_id"] == "dog_plush_n5_r16_ts42"
    assert record["broken_source_commit"] == "7138d20c19600663cfe8e7976fbc1ee460910dec"
    assert record["replacement_source_commit"] == "9efc6408bb262a469355b62d6e92ed3a6bad9c32"


def test_archive_cell_refuses_run_with_checkpoint(tmp_path):
    core_root = tmp_path / "core_v1"
    failed = make_failed_anchor(core_root)
    (failed / "trainer_output/checkpoint-1").mkdir()

    with pytest.raises(AssertionError, match="checkpoint"):
        run_archive_cell(core_root)

    assert failed.exists()
    assert not (core_root / "failed_attempts").exists()


def test_archive_cell_refuses_unexpected_failure(tmp_path):
    core_root = tmp_path / "core_v1"
    failed = make_failed_anchor(core_root, error="different failure")

    with pytest.raises(AssertionError, match="khác dự kiến"):
        run_archive_cell(core_root)

    assert failed.exists()
