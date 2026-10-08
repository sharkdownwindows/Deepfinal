# Core experiments trên Google Colab — Deepfinal

Cập nhật 08/10/2026: notebook này chạy **18 core training runs** bằng
`scripts/train_run.py` và registry của dự án. Upload
`notebooks/colab_train_deepfinal.ipynb` lên Colab, chọn GPU, chạy Cell 1 → 9.

| Phần | Thiết lập cho mỗi concept | Số run |
|---|---|---:|
| RQ1 — data size | n=1,3,5,10; rank=16 | 12 |
| RQ2 — rank bổ sung | n=5; rank=4,32 | 6 |
| Tổng | 3 concepts; n5/r16 dùng chung | 18 |

Concepts: `cat_mug`, `dog_plush`, `blue_white_vase`. Mọi run dùng SD1.5,
512×512, 500 updates, checkpoint mỗi 100 updates, training seed=42,
alpha=rank, text encoder frozen. Thông số lấy từ `configs/ml04_sweep.yaml`.
Prompt training lấy từ token/class trong config; evaluation prompt bank giữ nguyên.

Source: [Deepfinal](https://github.com/sharkdownwindows/Deepfinal), commit
`9efc6408bb262a469355b62d6e92ed3a6bad9c32`. Cell 8 chạy core anchor
`dog_plush_n5_r16_ts42` trước để kiểm tra train/reload trên GPU được cấp;
Cell 9 chạy các core runs còn thiếu, không train lại anchor hợp lệ.

Mỗi run hoàn tất training có resolved config, environment, provenance, adapter,
training log, metrics thời gian/VRAM và một ảnh kiểm tra reload do runner sinh.
Tên `adapter_pilot.png` trong runner chỉ là **ảnh kiểm tra reload của run 500 bước**;
không biến run core thành pilot training. Đây chưa phải batch evaluation 32 ảnh/mode.

Sau 18 runs vẫn cần sinh baseline/adapter cùng prompt/seed, DINO/CLIP và human eval
để hoàn tất nghiên cứu. Số training runs hoàn tất và số evaluation runs hoàn tất
là hai trạng thái khác nhau.

Chuẩn bị dataset v1 theo [DATA.md](../DATA.md). Vị trí mặc định là
`MyDrive/personalized-t2i/data`; nếu đã lưu chỗ khác, đặt `DATA_FOLDER` ở Cell 1.
Dataset phải được giải nén, với `raw/` và `manifests/` nằm ngay trong thư mục được chọn.
Notebook chỉ copy/kiểm tra 30 ảnh train;
9 ảnh held-out giữ riêng cho evaluation. Clone GitHub không tự tải dataset riêng.

Output mới: `MyDrive/personalized-t2i/core_v1/9efc6408bb26/artifacts/<run_id>/`.
Tách theo source commit để thiết kế cũ và run thất bại không bị ghi đè.
Các run từ `direct_training`, `colab_training_v2` hoặc pilot 256 không tự nhập
vào registry core. Một run train trực tiếp có thể được xét làm core nếu cấu hình
và bằng chứng đủ; tên thư mục tự nó không quyết định tính hợp lệ.

**Giới hạn hiện tại:** runner từ commit đã pin chưa có cờ resume.
Sau reset, notebook nhận diện và bỏ qua run completed hợp lệ; run bị ngắt
giữa chừng được giữ lại để xử lý checkpoint, không tự ghi đè hoặc train lại.
Chưa chạy GPU notebook này; cú pháp/test CPU không chứng minh training đã thành công.

## Cell 1 — Mount Drive, kiểm tra dữ liệu và GPU

Cell tìm Drive đã mount hoặc chọn mountpoint rỗng. Nó không xóa thư mục đang có
file và không tạo `MyDrive` giả. Chấp nhận yêu cầu kết nối Drive trên giao diện Colab.

`Mounted at ...` nghĩa là mount đã thành công. Nếu thiếu `concepts.csv`, cần chọn
đúng nơi dataset đã giải nén; đổi mountpoint hoặc tạo thư mục rỗng không bổ sung dữ liệu.
Mở Files bên trái Colab, tìm thư mục chứa đồng thời `raw/` và `manifests/`, copy path
vào `DATA_FOLDER` (hoặc nhập đường dẫn tương đối từ MyDrive). Nếu chỉ có ZIP, giải
nén dataset trước. `PERSIST` vẫn là nơi lưu kết quả; `DATA` có thể ở thư mục khác.
Để trống `DATA_FOLDER` sẽ tìm ở MyDrive, các thư mục con trực tiếp và các nhánh
`data/`, `data/data/` bên trong chúng. Không quét đệ quy toàn bộ Drive.

```python
from pathlib import Path
from google.colab import drive
import os
import subprocess
import sys

def choose_drive_path(content):
    mounted = sorted(p for p in content.iterdir()
                     if p.is_dir() and os.path.ismount(p) and (p / "MyDrive").is_dir())
    if mounted:
        return mounted[0]
    for i in range(100):
        candidate = content / f"lora_drive_{i}"
        if candidate.is_symlink():
            continue
        if not candidate.exists() or (candidate.is_dir() and not any(candidate.iterdir())):
            return candidate
    raise RuntimeError("Không tìm được mountpoint rỗng; kiểm tra các thư mục trong /content")

DRIVE = choose_drive_path(Path("/content"))
if not os.path.ismount(DRIVE):
    drive.mount(str(DRIVE))
assert os.path.ismount(DRIVE) and (DRIVE / "MyDrive").is_dir(), "Drive chưa mount thành công"
PERSIST = DRIVE / "MyDrive/personalized-t2i"
DATA_FOLDER = ""  # Điền đường dẫn dataset đã giải nén nếu lưu ở chỗ khác.

def find_data_root(my_drive, configured=""):
    if configured.strip():
        value = Path(configured.strip())
        candidates = [value if value.is_absolute() else my_drive / value]
    else:
        roots = [my_drive] + sorted(p for p in my_drive.iterdir() if p.is_dir())
        candidates = [p for root in roots for p in (root, root / "data", root / "data/data")]
    found = list(dict.fromkeys(p.resolve() for p in candidates
                 if (p / "manifests/concepts.csv").is_file() and (p / "raw").is_dir()))
    if len(found) == 1:
        return found[0]
    if len(found) > 1:
        raise ValueError("Có nhiều dataset; điền DATA_FOLDER để chọn đúng một thư mục:\n"
                         + "\n".join(str(p) for p in found))
    raise FileNotFoundError(
        "Drive đã mount, nhưng chưa thấy dataset đã giải nén.\n"
        "Điền DATA_FOLDER bằng thư mục chứa raw/ và manifests/concepts.csv.\n"
        "Nếu dataset nằm sâu hơn hoặc ngoài MyDrive, mở Files của Colab và copy path vào DATA_FOLDER.\n"
        "Nếu mới có ZIP, cần giải nén dataset trước."
    )

DATA = find_data_root(DRIVE / "MyDrive", DATA_FOLDER)
os.environ["HF_HOME"] = str(PERSIST / "hf_cache")
os.environ["HF_HUB_CACHE"] = str(PERSIST / "hf_cache/hub")
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
subprocess.run(["nvidia-smi"], check=True)
print("Drive:", DRIVE, "| data:", DATA)
```

## Cell 2 — Chuẩn bị thư viện

Các version dưới đây lấy từ môi trường notebook trước, chưa phải dependency lock
đã được xác nhận bằng training thành công. Giữ Torch/Torchvision có sẵn của Colab;
nếu dependency không tương thích thì pip phải báo lỗi. Không nâng Torch ngầm.
Cell gỡ riêng `torchao==0.10.0`, dependency tùy chọn đã gây lỗi PEFT trong notebook
trước; training này dùng AdamW, không dùng torchao quantization.

```python
from importlib.metadata import version, PackageNotFoundError

required = {"diffusers": "0.40.0", "transformers": "5.18.0",
            "accelerate": "1.15.0", "peft": "0.21.1"}
install = []
for package, wanted in required.items():
    try:
        current = version(package)
    except PackageNotFoundError:
        current = None
    if current != wanted:
        install.append(f"{package}=={wanted}")
for package in ("safetensors", "huggingface_hub", "PyYAML", "Pillow", "ftfy", "Jinja2", "tensorboard"):
    try:
        version(package)
    except PackageNotFoundError:
        install.append(package)

loaded = any(name in sys.modules for name in ("torch", "diffusers", "transformers", "peft"))
changed = False
if install:
    keep = [f"{name}=={version(name)}" for name in ("torch", "torchvision")]
    subprocess.run([sys.executable, "-m", "pip", "install", *install, *keep], check=True)
    changed = True
try:
    if version("torchao") == "0.10.0":
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=True)
        changed = True
except PackageNotFoundError:
    pass
if changed and loaded:
    raise RuntimeError("Đã đổi thư viện sau khi import. Restart Python session rồi chạy lại từ Cell 1")
print("Đã chuẩn bị dependencies; tiếp tục kiểm tra imports ở Cell 3")
```

## Cell 3 — Kiểm tra môi trường trước khi tải model

Kiểm tra trong process mới, có thử thêm LoRA vào một layer nhỏ trên CPU. Chưa tải
model ở bước này. HF_TOKEN là tùy chọn trong Colab Secrets nếu model yêu cầu quyền.

```python
subprocess.run([sys.executable, "-c", """
import torch, torchvision, diffusers, transformers, accelerate, peft
from peft import LoraConfig, get_peft_model
assert torch.cuda.is_available(), 'Chọn GPU trong Runtime settings'
model = torch.nn.Sequential(torch.nn.Linear(4, 4))
get_peft_model(model, LoraConfig(r=2, lora_alpha=2, target_modules=['0']))
print('Imports + PEFT adapter: OK; GPU:', torch.cuda.get_device_name(0))
"""], check=True)

import torch
import yaml
from google.colab import userdata
try:
    token = userdata.get("HF_TOKEN")
except (userdata.SecretNotFoundError, userdata.NotebookAccessError):
    token = None
if token:
    os.environ["HF_TOKEN"] = token
del token
```

## Cell 4 — Lấy đúng source, trainer và thiết lập package import

Giữ source cố định trong suốt experiment. Official trainer đặt trong checkout
Diffusers đúng commit để runner ghi được trainer SHA. Cell dùng PYTHONPATH cho
cả notebook và subprocess, xử lý lỗi `No module named personalized_t2i`.

Nếu notebook cũ báo `load_config_from_file() missing ... 'config_file'`, thay
khối Accelerate bằng bản dưới rồi chạy lại toàn bộ Cell 4. Accelerate 1.15.0
yêu cầu truyền `config_file`; writer và reader phải dùng cùng đường dẫn.
Không cần cài lại thư viện hay reset runtime chỉ để sửa lỗi gọi hàm này.

```python
import hashlib
from datetime import datetime, timezone

REPO_URL = "https://github.com/sharkdownwindows/Deepfinal.git"
SOURCE_COMMIT = "9efc6408bb262a469355b62d6e92ed3a6bad9c32"
REPO = Path("/content/deepfinal")
if not REPO.exists():
    subprocess.run(["git", "clone", "--no-checkout", REPO_URL, str(REPO)], check=True)
    subprocess.run(["git", "-C", str(REPO), "checkout", "--detach", SOURCE_COMMIT], check=True)

def git_value(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()

assert git_value(REPO, "remote", "get-url", "origin").removesuffix(".git") == REPO_URL.removesuffix(".git")
assert git_value(REPO, "rev-parse", "HEAD") == SOURCE_COMMIT, "Checkout khác SHA"
assert not git_value(REPO, "status", "--porcelain"), "Source có thay đổi; kiểm tra trước khi train"

TRAINER_SHA = "d035dcd7cc7c88e0a154609b62887d50bba9fdc2"
TRAINER_REPO = Path("/content/diffusers-core-v040")
if not TRAINER_REPO.exists():
    subprocess.run(["git", "clone", "--depth", "1", "--branch", "v0.40.0",
                    "https://github.com/huggingface/diffusers.git", str(TRAINER_REPO)], check=True)
assert git_value(TRAINER_REPO, "rev-parse", "HEAD") == TRAINER_SHA, "Trainer khác SHA đã pin"
assert not git_value(TRAINER_REPO, "status", "--porcelain"), "Trainer đã bị sửa"
TRAINER = TRAINER_REPO / "examples/dreambooth/train_dreambooth_lora.py"
subprocess.run([sys.executable, str(TRAINER), "--help"], check=True, stdout=subprocess.DEVNULL)

os.chdir(REPO)
sys.path[:0] = [str(REPO / "src"), str(REPO)]
os.environ["PYTHONPATH"] = os.pathsep.join(
    [str(REPO / "src"), str(REPO), os.environ.get("PYTHONPATH", "")]
)
from accelerate.utils import write_basic_config
from accelerate.commands.config.config_args import default_config_file, load_config_from_file
write_basic_config(mixed_precision="fp16", save_location=default_config_file)
accelerate_config = load_config_from_file(config_file=default_config_file)
assert accelerate_config.num_processes == 1 and accelerate_config.mixed_precision == "fp16"
assert accelerate_config.distributed_type == "NO", "Accelerate phải dùng một GPU/process"

from personalized_t2i.training.runner import validate_training_config
from scripts.run_sweep import expand_sweep_matrix, classify_existing_artifact

LEGACY_CORE_ROOT = PERSIST / "core_v1"
CORE_ROOT = LEGACY_CORE_ROOT / SOURCE_COMMIT[:12]
ARTIFACTS = CORE_ROOT / "artifacts"
SESSION = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
print("Core output:", ARTIFACTS, "| source:", SOURCE_COMMIT)
```

## Cell 5 — Kiểm tra 30 ảnh train và tạo đúng 18 cấu hình

Dùng manifest để chọn D1 ⊂ D3 ⊂ D5 ⊂ D10; runner sẽ materialize đúng subset cho
từng run. Copy raw images vào thư mục ignored của checkout Colab.
Không dùng `run_ml05_sweep.py` ở SHA này vì script đó chưa bao phủ ba concepts.
Sáu rank configs được suy ra từ anchor n5/r16 của cùng matrix.
Có thể chạy lại Cell 5 trong cùng session: config giống hệt được giữ nguyên;
config khác nội dung sẽ làm cell dừng, không ghi đè file cũ.

Runner hiện dùng crop mặc định của official trainer. Cell kiểm tra nguồn là ảnh
vuông sau EXIF; khi resize ảnh vuông về 512 rồi crop 512, không có dịch chuyển crop.
Nếu nguồn không vuông, dừng để thống nhất preprocessing trước khi chạy core.

```python
import copy
import csv
import json
import shutil
from PIL import Image, ImageOps
from personalized_t2i.config import CORE_CELLS

matrix = yaml.safe_load((REPO / "configs/ml04_sweep.yaml").read_text())
assert set(matrix["concepts"]) == {"cat_mug", "dog_plush", "blue_white_vase"}
assert matrix["training"]["resolution"] == 512
assert matrix["training"]["max_train_steps"] == 500
assert matrix["training"]["checkpointing_steps"] == 100
assert matrix["training"]["seed"] == 42

for concept_id, concept in matrix["concepts"].items():
    manifest = REPO / concept["manifest"]
    mirror = DATA / "manifests" / manifest.name
    assert hashlib.sha256(mirror.read_bytes()).digest() == hashlib.sha256(manifest.read_bytes()).digest()
    with manifest.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    train_rows = [r for r in rows if r["split"] == "train_pool"]
    assert len(train_rows) == 10 and all(r["concept_id"] == concept_id for r in rows)
    for row in train_rows:
        relative = Path(row["file_path"])
        assert not relative.is_absolute() and relative.parts[:2] == ("data", "raw")
        assert ".." not in relative.parts
        source = (DATA / Path(*relative.parts[1:])).resolve()
        assert source.is_relative_to(DATA.resolve())
        assert hashlib.sha256(source.read_bytes()).hexdigest() == row["sha256"].strip().lower()
        target = REPO / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            assert hashlib.sha256(target.read_bytes()).hexdigest() == row["sha256"].strip().lower()
        else:
            shutil.copy2(source, target)
        with Image.open(target) as image:
            oriented = ImageOps.exif_transpose(image)
            assert oriented.width == oriented.height, "Ảnh không vuông: cần thống nhất preprocessing"
    print(concept_id, ": 10 training images verified")

def make_core_configs(matrix, artifacts_root):
    configs = expand_sweep_matrix(matrix)
    anchors = [copy.deepcopy(c) for c in configs if c["data"]["subset_size"] == 5]
    for anchor in anchors:
        for rank in (4, 32):
            cfg = copy.deepcopy(anchor)
            cfg["training"].update(rank=rank, alpha=rank)
            cfg["run"]["id"] = (
                f"{cfg['data']['concept_id']}_n5_r{rank}_ts{cfg['training']['seed']}"
            )
            configs.append(cfg)
    for cfg in configs:
        cfg["run"]["artifact_namespace"] = "core"
        cfg["output"]["output_dir"] = str(artifacts_root / cfg["run"]["id"])
    expected = {(concept, n, rank) for concept in matrix["concepts"] for n, rank in CORE_CELLS}
    actual = {(c["data"]["concept_id"], c["data"]["subset_size"], c["training"]["rank"]) for c in configs}
    assert len(configs) == len(actual) == 18 and actual == expected
    assert len({c["run"]["id"] for c in configs}) == 18
    return configs

configs = make_core_configs(matrix, ARTIFACTS)
CONFIGS = {cfg["run"]["id"]: cfg for cfg in configs}
ANCHOR = "dog_plush_n5_r16_ts42"
RUN_IDS = [ANCHOR] + sorted(set(CONFIGS) - {ANCHOR})

# Configs cho session mới; không sửa config.resolved.yaml của run đã có.
CONFIG_DIR = CORE_ROOT / "configs" / SESSION
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
for run_id, cfg in CONFIGS.items():
    config_path = CONFIG_DIR / f"{run_id}.yaml"
    if config_path.exists():
        assert yaml.safe_load(config_path.read_text(encoding="utf-8")) == cfg, (
            f"Config khác nội dung; giữ nguyên để kiểm tra: {config_path}"
        )
    else:
        with config_path.open("x", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
print("Đúng 18 unique core configs; anchor chỉ xuất hiện một lần:", ANCHOR)
```

## Cell 6 — Khóa thiết kế, preflight và báo cáo trạng thái

Cell không train. So sánh config đầy đủ và provenance trước khi bỏ qua run cũ.
Thư mục tồn tại hoặc có weights riêng lẻ không được coi là completed.
Đổi mountpoint Drive không đổi danh tính run; vị trí thực tế được truyền lại theo
session. Source, trainer, package versions và thiết kế thí nghiệm vẫn phải khớp.

```python
def canonical_config(cfg):
    result = copy.deepcopy(cfg)
    result["output"]["output_dir"] = f"artifacts/{result['run']['id']}"
    return result

design = {
    "source_commit": SOURCE_COMMIT,
    "trainer_commit": TRAINER_SHA,
    "trainer_sha256": hashlib.sha256(TRAINER.read_bytes()).hexdigest(),
    "python_major_minor": list(sys.version_info[:2]),
    "packages": {name: version(name) for name in
                 ("torch", "torchvision", "diffusers", "transformers", "accelerate", "peft",
                  "numpy", "Pillow", "safetensors", "huggingface_hub")},
    "matrix_sha256": hashlib.sha256((REPO / "configs/ml04_sweep.yaml").read_bytes()).hexdigest(),
    "configs": [canonical_config(CONFIGS[run_id]) for run_id in sorted(CONFIGS)],
}
design_path = CORE_ROOT / "core_design.json"
if design_path.exists():
    assert json.loads(design_path.read_text()) == design, "Source/env/design khác core suite trước; cần kiểm tra"
else:
    with design_path.open("x", encoding="utf-8") as f:
        json.dump(design, f, indent=2)

def training_state(run_id):
    cfg = CONFIGS[run_id]
    run_dir = ARTIFACTS / run_id
    if not run_dir.exists():
        try:
            validate_training_config(cfg, REPO)
        except Exception as exc:
            return "preflight_failed", str(exc)
        return "ready", ""
    state, reason = classify_existing_artifact(run_dir, cfg, REPO)
    if state != "completed_existing":
        return "needs_review", reason
    try:
        saved = yaml.safe_load((run_dir / "config.resolved.yaml").read_text())
        assert canonical_config(saved) == canonical_config(cfg), "Resolved config khác"
        env = json.loads((run_dir / "environment.json").read_text())
        assert env["git"]["commit"] == SOURCE_COMMIT and env["git"]["dirty"] is False
        assert env["diffusers"]["git_commit"] == TRAINER_SHA
        for package in ("torch", "diffusers", "transformers", "accelerate", "peft"):
            assert env["packages"][package]["version"] == design["packages"][package]
        metrics = json.loads((run_dir / "metrics.json").read_text())
        assert metrics["run_id"] == run_id and metrics["training_steps"] == 500
        for field in ("adapter_path", "generation_path"):
            path = (run_dir / metrics[field]).resolve()
            assert path.is_relative_to(run_dir.resolve()) and path.is_file() and path.stat().st_size > 0
        assert (run_dir / "logs/train.log").is_file()
        subset = json.loads((run_dir / "training_subset.json").read_text())
        assert len(subset) == cfg["data"]["subset_size"]
        records = [json.loads(line) for line in (run_dir / "pilot_metadata.jsonl").read_text().splitlines() if line]
        assert len(records) == 1 and records[0]["run_id"] == run_id
        assert records[0]["generation_mode"] == "adapter"
        assert records[0]["width"] == records[0]["height"] == 512
    except Exception as exc:
        return "needs_review", f"Artifact validation: {type(exc).__name__}: {exc}"
    return "completed_training", "Train + reload smoke verified; evaluation pending"

def report_core():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    rows = []
    for run_id in RUN_IDS:
        state, reason = training_state(run_id)
        rows.append({"run_id": run_id, "training_status": state, "reason": reason,
                     "evaluation_status": "not_checked"})
        print(f"{state:20} {run_id} {reason}")
    path = CORE_ROOT / f"training-status-{stamp}.csv"
    with path.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    count = sum(r["training_status"] == "completed_training" for r in rows)
    print(f"Core training verified: {count}/18. Evaluation: chưa kiểm tra ở notebook này.")
    return rows

rows = report_core()
assert not any(r["training_status"] == "preflight_failed" for r in rows), "Sửa lỗi preflight trước"
```

## Cell 7 — Archive đúng run anchor thất bại cũ

Cell này chỉ xử lý run `dog_plush_n5_r16_ts42` của source cũ đã thất bại ở
step 0 vì thiếu đối số mixed precision cho trainer. Nó kiểm tra status, thông báo
lỗi và xác nhận không có checkpoint trước khi **move** nguyên thư mục sang
`core_v1/failed_attempts/`. Không xóa file, không dùng glob và không đụng run khác.
Sau khi archive, suite mới vẫn ghi dưới thư mục source commit riêng.

```python
BROKEN_SOURCE_COMMIT = "7138d20c19600663cfe8e7976fbc1ee460910dec"
FAILED_RUN_DIR = LEGACY_CORE_ROOT / "artifacts" / ANCHOR
if FAILED_RUN_DIR.exists():
    status_path = FAILED_RUN_DIR / "status.json"
    assert status_path.is_file(), f"Thiếu status, không tự move: {status_path}"
    failed_status = json.loads(status_path.read_text(encoding="utf-8"))
    assert failed_status.get("run_id") == ANCHOR
    assert failed_status.get("status") == "failed", "Chỉ archive run có status=failed"
    assert failed_status.get("error") == "Diffusers trainer exited with code 1", (
        f"Lỗi khác dự kiến, giữ nguyên để kiểm tra: {failed_status.get('error')}"
    )
    old_trainer_output = FAILED_RUN_DIR / "trainer_output"
    checkpoints = [] if not old_trainer_output.is_dir() else [
        path for path in old_trainer_output.iterdir()
        if path.is_dir() and path.name.startswith("checkpoint-")
    ]
    assert not checkpoints, f"Run có checkpoint, không tự move: {checkpoints}"

    archived_at = datetime.now(timezone.utc)
    archive_dir = (
        LEGACY_CORE_ROOT / "failed_attempts" /
        f"{ANCHOR}-{BROKEN_SOURCE_COMMIT[:12]}-{archived_at.strftime('%Y%m%dT%H%M%S%fZ')}"
    )
    archive_dir.parent.mkdir(parents=True, exist_ok=True)
    assert not archive_dir.exists()
    FAILED_RUN_DIR.rename(archive_dir)
    record = {
        "run_id": ANCHOR,
        "old_path": str(FAILED_RUN_DIR),
        "archive_path": str(archive_dir),
        "archived_at": archived_at.isoformat(),
        "reason": "fp16 precision was passed to accelerate launcher but not to trainer script",
        "broken_source_commit": BROKEN_SOURCE_COMMIT,
        "replacement_source_commit": SOURCE_COMMIT,
        "original_status": failed_status,
    }
    with (archive_dir / "archive_record.json").open("x", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    print("Archived failed attempt:", archive_dir)
else:
    print("Không còn failed anchor ở đường dẫn cũ; không có gì để move.")
```

## Cell 8 — Chạy core anchor 500 bước và kiểm tra reload

Đây là **một trong 18 core runs**, được chạy trước để xác nhận compute.
Runner tự lưu environment/config/provenance, training subset, checkpoint và
adapter rồi reload sinh một ảnh kiểm tra. Chưa thay bank 8 prompts × 4 seeds.
Log/progress xuất ở output của cell.

```python
import signal
import time

def run_one_core(run_id):
    state, reason = training_state(run_id)
    if state == "completed_training":
        print("Skip verified completed:", run_id)
        return
    if state != "ready":
        raise RuntimeError(f"{run_id}: {state}: {reason}. Giữ nguyên artifact để xử lý.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    command = [sys.executable, str(REPO / "scripts/train_run.py"),
               "--config", str(CONFIG_DIR / f"{run_id}.yaml"), "--diffusers-script", str(TRAINER)]
    invocation_dir = CORE_ROOT / "invocations"
    invocation_dir.mkdir(exist_ok=True)
    record = {"run_id": run_id, "command": command, "returncode": None, "state": "starting"}
    process = None
    started = time.monotonic()
    try:
        with (invocation_dir / f"{run_id}-{stamp}.log").open("x", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=str(REPO), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
            record["state"] = "running"
            for line in process.stdout:
                print(line, end="")
                log.write(line)
                log.flush()
            record["returncode"] = process.wait()
        assert record["returncode"] == 0, "Training/reload lỗi; xem log và status.json của run"
        state, reason = training_state(run_id)
        assert state == "completed_training", f"{state}: {reason}"
        record["state"] = state
    except BaseException as exc:
        record.update(state="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                      error=type(exc).__name__)
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        raise
    finally:
        record["wall_seconds"] = time.monotonic() - started
        with (invocation_dir / f"{run_id}-{stamp}.json").open("x", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

run_one_core(ANCHOR)
anchor_metrics = json.loads((ARTIFACTS / ANCHOR / "metrics.json").read_text())
print("Anchor training seconds:", anchor_metrics["training_wall_seconds"])
print("Observed GPU memory MiB:", anchor_metrics["observed_peak_gpu_memory_used_mib"])
print("Số đo nvidia-smi có thể gồm tiến trình khác; không phải peak tensor allocation riêng.")
```

## Cell 9 — Chạy phần còn lại của core matrix

Sau anchor thành công, chạy cell này để xử lý các ô còn thiếu. Run completed được
kiểm tra rồi bỏ qua. Run dở/failed đã tồn tại được báo riêng và giữ nguyên; notebook
không có chức năng resume chúng. Nếu run mới lỗi, cell dừng ở traceback để xử lý
nguyên nhân. Báo cáo cuối vẫn liệt kê toàn bộ 18 ô.

```python
assert training_state(ANCHOR)[0] == "completed_training", "Anchor chưa vượt qua train/reload"
try:
    for run_id in RUN_IDS:
        state, reason = training_state(run_id)
        if state == "completed_training":
            print("Skip:", run_id)
        elif state == "ready":
            print("Start core run:", run_id)
            run_one_core(run_id)
        else:
            print("Giữ lại để xử lý:", run_id, state, reason)
finally:
    rows = report_core()

completed = sum(r["training_status"] == "completed_training" for r in rows)
if completed == 18:
    print("Đã xác minh 18/18 core training runs. Bước tiếp theo: generation/evaluation.")
else:
    print(f"Mới xác minh {completed}/18; các ô còn lại chưa hoàn tất.")
```

## Kết quả nằm ở đâu?

Trong `MyDrive/personalized-t2i/core_v1/9efc6408bb26/`:

```text
core_design.json
configs/<session>/<run_id>.yaml
training-status-<timestamp>.csv
invocations/<run_id>-<timestamp>.log
artifacts/<run_id>/
  status.json
  config.resolved.yaml
  environment.json
  provenance.json
  training_subset.json
  metrics.json
  logs/train.log
  trainer_output/checkpoint-100/
  trainer_output/checkpoint-200/ ...
  adapter/pytorch_lora_weights.safetensors
  generations/adapter_pilot.png
  pilot_metadata.jsonl
```

Tên checkpoint thực tế nằm trong `trainer_output/`; thư mục `checkpoints/` do
registry tạo có thể trống. Đừng suy ra không có checkpoint từ thư mục trống đó.

- Runtime reset giữa **hai run**: chạy Cell 1 → 9 lại; notebook skip các run completed
  hợp lệ và tiếp tục những ô chưa bắt đầu.
- Runtime bị ngắt **giữa một run**: checkpoint còn trên Drive; runner hiện từ chối
  output directory đã có. Cần xử lý resume bằng checkpoint/config cũ hoặc một
  attempt mới có provenance trước khi nghiệm thu ô đó. Notebook chưa giải quyết
  tự động tình huống này.
- Nếu GPU OOM, giữ log và điều chỉnh cách dùng bộ nhớ nhất quán trước khi tiếp tục;
  giảm resolution xuống 256 sẽ thay đổi thí nghiệm.
- Khi 18/18 training xong, dùng fixed bank 8 prompts × 4 seeds để sinh baseline
  và adapter, rồi chạy DINO/CLIP trên held-out references. `status=completed`
  của training runner không chứng minh toàn bộ evaluation/report đã xong.
- Mọi image/weight/log/runtime paths nằm trong Colab và Drive; notebook không push
  dữ liệu hoặc model lên GitHub.

Nguồn: [training matrix](../configs/ml04_sweep.yaml),
[project runner](../src/personalized_t2i/training/runner.py),
[official trainer pinned](https://github.com/huggingface/diffusers/blob/d035dcd7cc7c88e0a154609b62887d50bba9fdc2/examples/dreambooth/train_dreambooth_lora.py),
[Colab FAQ](https://research.google.com/colaboratory/faq.html).

