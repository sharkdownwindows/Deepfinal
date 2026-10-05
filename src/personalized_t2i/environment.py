from pathlib import Path

import yaml


REQUIRED_PACKAGES = [
    "torch",
    "diffusers",
    "transformers",
    "peft",
    "accelerate",
]


def validate_environment_metadata(data):
    if not isinstance(data, dict):
        raise ValueError("Environment metadata must be a dictionary.")

    if not isinstance(data.get("git"), dict):
        raise ValueError("Missing git information.")

    if not isinstance(data["git"].get("commit"), str):
        raise ValueError("git.commit must be a string.")

    if not isinstance(data.get("python"), dict):
        raise ValueError("Missing python information.")

    if not isinstance(data["python"].get("version"), str):
        raise ValueError("python.version must be a string.")

    packages = data.get("packages")

    if not isinstance(packages, dict):
        raise ValueError("packages must be a dictionary.")

    for package in REQUIRED_PACKAGES:
        if package not in packages:
            raise ValueError(f"Missing package: {package}")

        if not isinstance(packages[package], dict):
            raise ValueError(f"packages.{package} must be a dictionary.")

        if not isinstance(packages[package].get("version"), str):
            raise ValueError(
                f"packages.{package}.version must be a string."
            )

    cuda = data.get("cuda")

    if not isinstance(cuda, dict):
        raise ValueError("Missing cuda information.")

    cuda_available = cuda.get("available")

    if not isinstance(cuda_available, bool):
        raise ValueError("cuda.available must be a boolean.")

    cuda_version = cuda.get("version")

    if cuda_available:
        if not isinstance(cuda_version, str):
            raise ValueError(
                "cuda.version must be a string when CUDA is available."
            )
    else:
        if cuda_version is not None:
            raise ValueError(
                "cuda.version must be null when CUDA is unavailable."
            )

    driver = data.get("driver")

    if not isinstance(driver, dict):
        raise ValueError("Missing driver information.")

    driver_version = driver.get("version")

    if driver_version is not None and not isinstance(driver_version, str):
        raise ValueError("driver.version must be a string or null.")

    gpu = data.get("gpu")

    if not isinstance(gpu, dict):
        raise ValueError("Missing gpu information.")

    if gpu.get("name") is not None and not isinstance(gpu.get("name"), str):
        raise ValueError("gpu.name must be a string or null.")

    gpu_count = gpu.get("count")

    if gpu_count is not None:
        if type(gpu_count) is not int:
            raise ValueError("gpu.count must be an integer or null.")
        if gpu_count < 0:
            raise ValueError("gpu.count must be greater than or equal to 0.")

    gpu_vram_mb = gpu.get("vram_mb")

    if gpu_vram_mb is not None:
        if type(gpu_vram_mb) is not int:
            raise ValueError("gpu.vram_mb must be an integer or null.")
        if gpu_vram_mb < 0:
            raise ValueError(
                "gpu.vram_mb must be greater than or equal to 0."
            )

    platform = data.get("platform")

    if not isinstance(platform, dict):
        raise ValueError("Missing platform information.")

    for field in ["system", "release", "machine"]:
        if not isinstance(platform.get(field), str):
            raise ValueError(f"platform.{field} must be a string.")

    return True


def validate_environment_file(path):
    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    return validate_environment_metadata(data)