import pytest

from personalized_t2i.training.train import prepare_run
from tests.test_config import valid_config


def test_prepare_run_creates_artifacts(tmp_path):
    config = valid_config()
    config_path = tmp_path / "config.yaml"

    import yaml

    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)

    environment = {"test": True}

    run_dir = prepare_run(
        config_path=config_path,
        environment=environment,
        artifacts_root=tmp_path / "artifacts",
    )

    assert run_dir.exists()
    assert (run_dir / "config.resolved.yaml").exists()
    assert (run_dir / "environment.json").exists()
    assert (run_dir / "status.json").exists()


def test_prepare_run_rejects_duplicate_run(tmp_path):
    config = valid_config()
    config_path = tmp_path / "config.yaml"

    import yaml

    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)

    environment = {"test": True}
    artifacts_root = tmp_path / "artifacts"

    prepare_run(
        config_path=config_path,
        environment=environment,
        artifacts_root=artifacts_root,
    )

    with pytest.raises(FileExistsError):
        prepare_run(
            config_path=config_path,
            environment=environment,
            artifacts_root=artifacts_root,
        )
