# Colab sau reset: chỉ các cell cần để train LoRA

Với run mới trên repo **Deepfinal**, dùng [hướng dẫn Colab mới](COLAB_TRAIN_DEEPFINAL.md)
và [notebook 9 cell](../notebooks/colab_train_deepfinal.ipynb). Bản dưới đây giữ lại
để đối chiếu các run cũ trong `direct_training`; không dùng lẫn checkpoint/config
giữa hai hướng dẫn.

Notebook đã kiểm tra ngày 07/10/2026:
https://colab.research.google.com/drive/1U6DYFp7VcZ6cgZI4o5RB8PJCAZDS5xVW

Chạy lần lượt **Cell 1 → 7** dưới đây thay cho Run all notebook cũ.
Cell 7 có thể tiếp tục từ checkpoint đã lưu trên Drive. Nếu chưa từng lưu
checkpoint, không thể khôi phục các steps đã mất khi runtime reset.

Luồng này train trực tiếp bằng official Diffusers trainer, lấy hyperparameters
từ YAML trong repo và ghi resolved config/checkpoints/log lên Drive. Nó không
chạy CLIP/DINO, batch evaluation hay Gradio. Đây là training riêng, **không tự
nghiệm thu issues #7/#9/#14/#15 hoặc tự nhập kết quả vào registry core**.
Wrapper trong repo hiện vẫn đang được Desktop sửa; không trộn artifact của
luồng này với pilot 1 step hoặc các core runs khác.

## Những cell cũ bỏ khỏi luồng train

| Cell cũ nhận diện bằng nội dung | Xử lý |
|---|---|
| `snapshot_download(MODEL_ID, revision=...)` không có filter | Bỏ. Output cũ cho thấy đã tải khoảng 44–47 GB cả repository model |
| Load `StableDiffusionPipeline` và sinh ảnh vase thử | Bỏ trước training; tránh giữ thêm pipeline trên GPU |
| `whoami()` | Bỏ |
| `!ls /content`, in CUDA nhiều lần | Gộp thành Cell 1 và 3 |
| Tạo thư mục dataset rỗng | Bỏ; thư mục tồn tại không chứng minh có dữ liệu |
| Copy raw/eval_refs nhiều lần, dùng lẫn drive/gdrive | Thay bằng Cell 5: chỉ copy đúng subset cần train |
| Preprocess lại cả 30 ảnh | Bỏ khỏi recovery; official trainer resize/crop ảnh train |
| Tạo 18 YAML và tất cả D1/D3/D5/D10 | Bỏ khi chỉ chạy một run; Cell 5 tạo một config/subset |
| In mọi YAML, in hash từng manifest riêng | Thay bằng kiểm tra hash của subset tại Cell 5 |
| `git diff` egg-info ở anchor `4dLk6FR42S6I` | Bỏ; không phục vụ training |
| `git restore`, tạo branch, fetch/pull nhiều lần | Bỏ khỏi recovery; không thay source giữa một run |
| Cài editable project nhiều lần, sửa sys.path nhiều lần | Bỏ; luồng này không import package project |
| CLIP/DINO, prompt bank evaluation, charts/demo | Làm sau khi train xong, không cần tải metric models lúc này |

## Cell 1 — Mount Drive và kiểm tra GPU

Chọn GPU trong Runtime settings trước. Không tạo `MyDrive` giả bằng `mkdir`.

```python
from pathlib import Path
from google.colab import drive
import os
import subprocess
import sys

mounted = [Path(p) for p in ("/content/gdrive", "/content/drive") if os.path.ismount(p)]
if mounted:
    DRIVE = mounted[0]
else:
    DRIVE = Path("/content/gdrive")
    if DRIVE.exists() and any(DRIVE.iterdir()):
        DRIVE = Path("/content/lora_drive")
    drive.mount(str(DRIVE))

assert os.path.ismount(DRIVE), "Google Drive chưa mount thật"
PERSIST = DRIVE / "MyDrive/personalized-t2i"
DATA = PERSIST / "data"
assert (DATA / "manifests/concepts.csv").is_file(), (
    f"Thiếu {DATA}/manifests/concepts.csv. Upload thư mục data đã có; không tạo thư mục rỗng."
)

# Thiết lập trước khi import huggingface_hub/diffusers.
# Những lần reset VM sau sẽ dùng lại cache được lưu ở Drive.
os.environ["HF_HOME"] = str(PERSIST / "hf_cache")
os.environ["HF_HUB_CACHE"] = str(PERSIST / "hf_cache/hub")
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
subprocess.run(["nvidia-smi"], check=True)
print("Persistent data:", DATA)
```

Nếu DATA thiếu, upload **nội dung** `D:\AI thực chiến\Lecture Vinuni\data\data`
vào `MyDrive/personalized-t2i/data/`. Không lồng thêm một thư mục `data`.

## Cell 2 — Cài dependencies còn thiếu

Giữ Torch/Torchvision do Colab cung cấp. Các version dưới đây được lấy từ saved
environment của notebook; chưa có bằng chứng training thành công với bộ này.
Không chạy `pip install -U torch`, không cài toàn bộ requirements project.

```python
from importlib.metadata import version, PackageNotFoundError

required = {
    "diffusers": "0.40.0",
    "transformers": "5.18.0",
    "accelerate": "1.15.0",
    "peft": "0.21.1",
}
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
if install:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *install], check=True)
else:
    print("Dependencies đã có: bỏ qua pip install")
```

`bitsandbytes` không cần vì dùng AdamW mặc định. Nếu đã import phiên bản khác
trước Cell 2 và pip thay nó, restart **Python session** rồi chạy lại từ Cell 1;
không cần chủ động disconnect/delete runtime. Chạy các cell này ngay từ đầu
sẽ tránh import package cũ trước khi cài.

## Cell 3 — Kiểm tra imports; HF token nếu cần

```python
import torch
import torchvision
import diffusers
import transformers
import accelerate
import peft
import yaml

assert torch.cuda.is_available(), "Chưa có CUDA GPU; không train trên CPU"
print(torch.cuda.get_device_name(0))
print("Diffusers:", diffusers.__version__)

# Dùng secret nếu bạn đã khai báo HF_TOKEN trong Colab Secrets.
# Public model có thể được truy cập không cần đăng nhập; không in token/whoami.
from google.colab import userdata
try:
    token = userdata.get("HF_TOKEN")
except (userdata.SecretNotFoundError, userdata.NotebookAccessError):
    token = None
if token:
    os.environ["HF_TOKEN"] = token
```

Nếu import lỗi, dừng và sửa dependency trước. Không xóa version check của trainer.

### Nếu lỗi `Found an incompatible version of torchao ... 0.10.0`

PEFT 0.21.1 phát hiện torchao cũ trong Colab khi thêm LoRA adapter. Luồng này
không dùng torchao quantization, nên gỡ dependency tùy chọn này thay vì nâng
Torch hoặc thay đổi model. Chạy một cell:

```python
%pip uninstall -y torchao
```

Xác minh trong process mới (không dùng kết quả import cache trong kernel):

```python
subprocess.run([
    sys.executable, "-c",
    "from peft.import_utils import is_torchao_available; "
    "assert not is_torchao_available(); print('torchao optional backend: disabled')",
], check=True)
```

Sau đó chạy lại cell train. Không cần reset runtime hoặc xóa cache/model/data.
Trainer được khởi động bằng process mới nên không dùng module cache của kernel.

## Cell 4 — Lấy repo và official trainer đúng phiên bản

Repo chỉ clone khi thiếu. Nếu `/content/project` còn tồn tại sau reset kernel,
giữ nguyên checkout; không pull/reset hoặc clone đè.

```python
REPO = Path("/content/project")
if not REPO.exists():
    subprocess.run([
        "git", "clone", "--depth", "1",
        "https://github.com/sharkdownwindows/Personalized-Text-to-Image-Generation-with-LoRA.git",
        str(REPO),
    ], check=True)
assert (REPO / "configs/pilot/dog_plush_n10_r16_ts42.yaml").is_file()
SOURCE_COMMIT = subprocess.check_output(
    ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
).strip()

# Official v0.40.0: commit được xác minh từ tag; không dùng trainer trong repo
# đang yêu cầu diffusers 0.41.0.dev0 với environment 0.40.0.
TRAINER_SHA = "d035dcd7cc7c88e0a154609b62887d50bba9fdc2"
TRAINER = PERSIST / "trainer-v0.40.0/train_dreambooth_lora.py"
TRAINER.parent.mkdir(parents=True, exist_ok=True)
if not TRAINER.is_file():
    import urllib.request
    urllib.request.urlretrieve(
        f"https://raw.githubusercontent.com/huggingface/diffusers/{TRAINER_SHA}"
        "/examples/dreambooth/train_dreambooth_lora.py",
        TRAINER,
    )
subprocess.run([sys.executable, str(TRAINER), "--help"], check=True, stdout=subprocess.DEVNULL)
print("Project commit:", SOURCE_COMMIT)
print("Trainer:", TRAINER)
```

## Cell 5 — Chọn một run và chuẩn bị đúng subset

Chỉ sửa `CONCEPT`, `N`, `RANK` để chọn run. Mặc định anchor `dog_plush`, n5,
rank16, seed42, 500 updates. Subsets chọn theo manifest và kiểm tra hash.
Không cần chạy lại preprocessing/sinh toàn bộ 18 YAML của notebook cũ.

```python
import csv
import hashlib
import shutil
import json

CONCEPT, N, RANK = "dog_plush", 5, 16
assert (N, RANK) in {(1,16), (3,16), (5,16), (10,16), (5,4), (5,32)}
RUN_ID = f"{CONCEPT}_n{N}_r{RANK}_ts42"

with (DATA / "manifests/concepts.csv").open(encoding="utf-8", newline="") as f:
    concepts = {r["concept_id"]: r for r in csv.DictReader(f)}
concept = concepts[CONCEPT]
manifest = DATA / "manifests" / f"{CONCEPT}_v1.csv"
with manifest.open(encoding="utf-8", newline="") as f:
    selected = [r for r in csv.DictReader(f) if r["split"] == "train_pool"
                and str(N) in r["subset_membership"].split(",")]
assert len(selected) == N
TRAIN_DATA = Path("/content/lora_subsets") / RUN_ID
TRAIN_DATA.mkdir(parents=True, exist_ok=True)
expected_names = {Path(r["file_path"]).name for r in selected}
assert not (set(p.name for p in TRAIN_DATA.iterdir()) - expected_names), "Subset có file thừa"
for row in selected:
    relative = Path(row["file_path"])
    assert relative.parts[0] == "data" and ".." not in relative.parts
    source = DATA.parent / relative
    assert hashlib.sha256(source.read_bytes()).hexdigest() == row["sha256"], source
    target = TRAIN_DATA / source.name
    if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest() != row["sha256"]:
        shutil.copy2(source, target)

cfg = yaml.safe_load((REPO / "configs/pilot/dog_plush_n10_r16_ts42.yaml").read_text())
cfg["run"] = {"id": RUN_ID, "protocol_version": "v1", "artifact_namespace": "direct_training"}
cfg["data"].update(concept_id=CONCEPT, dataset_version="v1", subset_size=N,
    manifest=str(manifest), manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
    train_data_dir=str(TRAIN_DATA),
    instance_prompt=f"a photo of {concept['unique_token']} {concept['class_noun']}")
cfg["training"].update(rank=RANK, alpha=RANK, max_train_steps=500,
    checkpointing_steps=100, seed=42, train_text_encoder=False)
cfg["inference"]["prompt"] = cfg["data"]["instance_prompt"] + " on a wooden table"
cfg["inference"]["num_inference_steps"] = 30
RUN_DIR = PERSIST / "direct_training" / RUN_ID
cfg["output"]["output_dir"] = str(RUN_DIR)

# Không ghi đè config cũ hoặc resume một thí nghiệm khác.
CONFIG = RUN_DIR / "config.resolved.yaml"
if CONFIG.exists():
    assert yaml.safe_load(CONFIG.read_text()) == cfg, "Config cũ khác: dừng, không resume"
else:
    assert not RUN_DIR.exists() or not any(RUN_DIR.iterdir()), "Artifact có sẵn nhưng thiếu config"
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(RUN_ID, "| images:", len(selected), "| output:", RUN_DIR)
```

## Cell 6 — Lưu environment và cấu hình Accelerate

```python
from accelerate.utils import write_basic_config
write_basic_config(mixed_precision="fp16")
evidence = {
    "project_commit": SOURCE_COMMIT, "trainer_commit": TRAINER_SHA,
    "trainer_sha256": hashlib.sha256(TRAINER.read_bytes()).hexdigest(),
    "gpu": torch.cuda.get_device_name(0), "python": sys.version,
    "torch": torch.__version__, "diffusers": diffusers.__version__,
    "transformers": transformers.__version__, "accelerate": accelerate.__version__,
    "peft": peft.__version__, "execution": "direct official trainer; not project registry",
}
# Mỗi session giữ evidence riêng, không sửa evidence session trước.
from datetime import datetime, timezone
SESSION = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
(RUN_DIR / f"environment-{SESSION}.json").write_text(json.dumps(evidence, indent=2))
with (RUN_DIR / f"pip-freeze-{SESSION}.txt").open("w") as f:
    subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=f, check=True)
```

## Cell 7 — Train hoặc resume checkpoint

Model được trainer tải từng component cần dùng; không chạy `snapshot_download()`
toàn repository trước. Cache mới nằm trên Drive nên sống qua VM reset. Nếu cache
cũ chỉ nằm trong VM đã bị xóa thì lần đầu vẫn phải tải lại weights cần thiết.
Drive I/O có thể chậm hơn local disk; ảnh train nhỏ được copy local ở Cell 5.

```python
train = cfg["training"]
final_weights = RUN_DIR / "pytorch_lora_weights.safetensors"
assert not final_weights.exists(), (
    "Đã có final weights: không train lại. Kiểm tra kết quả hoặc chọn run khác ở Cell 5."
)
command = [sys.executable, "-m", "accelerate.commands.launch",
    "--num_processes", "1", "--num_machines", "1", "--mixed_precision", "fp16",
    str(TRAINER),
    "--pretrained_model_name_or_path", cfg["model"]["id"],
    "--revision", cfg["model"]["revision"],
    "--instance_data_dir", str(TRAIN_DATA),
    "--instance_prompt", cfg["data"]["instance_prompt"],
    "--output_dir", str(RUN_DIR),
    "--resolution", str(train["resolution"]),
    "--train_batch_size", str(train["batch_size"]),
    "--gradient_accumulation_steps", str(train["gradient_accumulation_steps"]),
    "--learning_rate", str(train["learning_rate"]),
    "--lr_scheduler", train["scheduler"],
    "--lr_warmup_steps", str(train.get("warmup_steps", 0)),
    "--max_train_steps", str(train["max_train_steps"]),
    "--checkpointing_steps", str(train["checkpointing_steps"]),
    "--seed", str(train["seed"]), "--rank", str(train["rank"]),
    "--gradient_checkpointing", "--report_to", "tensorboard"]
checkpoints = sorted(
    [p for p in RUN_DIR.glob("checkpoint-*") if p.is_dir() and p.name.split("-")[-1].isdigit()],
    key=lambda p: int(p.name.split("-")[-1]),
)
if checkpoints:
    # Chỉ resume checkpoint Accelerate đầy đủ, không phải chỉ file adapter.
    latest = checkpoints[-1]
    assert any(latest.glob("optimizer*")) and any(latest.glob("random_states*")), (
        f"Checkpoint có thể chưa ghi xong: {latest}. Kiểm tra trước khi resume."
    )
    command += ["--resume_from_checkpoint", str(latest)]
    print("Resume:", latest.name)
else:
    print("Train từ step 0; checkpoint mới mỗi 100 updates")

import time
started = time.monotonic()
with (RUN_DIR / f"train-{SESSION}.log").open("w") as log:
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1)
    for line in process.stdout:
        print(line, end="")
        log.write(line)
        log.flush()
    code = process.wait()
(RUN_DIR / f"session-{SESSION}.json").write_text(json.dumps({
    "returncode": code, "session_wall_seconds": time.monotonic() - started,
    "resumed_from": str(checkpoints[-1]) if checkpoints else None,
}, indent=2))
assert code == 0, f"Trainer lỗi {code}; xem log trong {RUN_DIR}"
assert final_weights.is_file(), "Trainer không tạo final LoRA weights"
print("Đã lưu final LoRA weights:", final_weights)
```

Checkpoint, final adapter, config và log được ghi trực tiếp vào
`MyDrive/personalized-t2i/direct_training/<run_id>/`. Không cần đợi cuối run mới
copy checkpoint. Nếu bị ngắt giữa lúc ghi checkpoint, kiểm tra checkpoint đầy
đủ trước khi resume; không xóa artifact lỗi hoặc tự coi run là completed.

## Sau khi train xong

- Kiểm tra adapter load/inference, rồi mới chạy evaluation bằng pipeline đã sửa.
- Direct trainer không xuất đầy đủ artifact contract của repo và không tự đo
  peak VRAM; không dùng riêng final weights để đóng issues #7/#9/#14.
- Khi đưa run này vào core evidence, giữ trainer/environment/config/subset
  hashes và kiểm tra preprocessing tương thích các run khác trước khi so sánh.
- Chỉ thay Cell 5 để chọn concept/n/rank khác; giữ model revision, seed,
  training steps và environment cố định. Không train lại anchor đã hoàn tất.

## Giới hạn và nguồn

Các cell được viết dựa trên notebook đã đọc và CLI official trainer v0.40.0.
Đã xác minh tag/commit, version gate và cấu trúc cell; **chưa chạy GPU các cell
này**, chưa chứng minh compatibility hoặc tốc độ. Không có con số runtime dự đoán.

- [Hugging Face downloads/cache](https://huggingface.co/docs/huggingface_hub/guides/download)
- [Official trainer đã pin](https://github.com/huggingface/diffusers/blob/d035dcd7cc7c88e0a154609b62887d50bba9fdc2/examples/dreambooth/train_dreambooth_lora.py)
- [Colab runtime và lưu trữ](https://research.google.com/colaboratory/faq.html)
