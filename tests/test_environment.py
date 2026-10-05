import pytest

from personalized_t2i.environment import validate_environment_metadata


def base_environment():
    return {
        "git": {
            "commit": "example-commit",
        },
        "python": {
            "version": "3.x",
        },
        "packages": {
            "torch": {
                "version": "example",
            },
            "diffusers": {
                "version": "example",
            },
            "transformers": {
                "version": "example",
            },
            "peft": {
                "version": "example",
            },
            "accelerate": {
                "version": "example",
            },
        },
        "cuda": {
            "available": False,
            "version": None,
        },
        "driver": {
            "version": None,
        },
        "gpu": {
            "name": None,
            "count": None,
            "vram_mb": None,
        },
        "platform": {
            "system": "Windows",
            "release": "example",
            "machine": "example",
        },
    }


def test_valid_cpu_environment():
    data = base_environment()

    assert validate_environment_metadata(data) is True


def test_valid_gpu_environment():
    data = base_environment()

    data["cuda"]["available"] = True
    data["cuda"]["version"] = "12.x"
    data["driver"]["version"] = "example"
    data["gpu"]["name"] = "Example NVIDIA GPU"
    data["gpu"]["count"] = 1
    data["gpu"]["vram_mb"] = 0

    assert validate_environment_metadata(data) is True


def test_cpu_cannot_have_cuda_version():
    data = base_environment()

    data["cuda"]["version"] = "12.x"

    with pytest.raises(ValueError):
        validate_environment_metadata(data)


def test_gpu_requires_cuda_version():
    data = base_environment()

    data["cuda"]["available"] = True

    with pytest.raises(ValueError):
        validate_environment_metadata(data)