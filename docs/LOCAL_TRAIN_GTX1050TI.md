# Train LoRA trên máy Windows này

Đã kiểm tra ngày 2026-10-07: GTX 1050 Ti 4 GB, RAM khoảng 16 GB,
VRAM trống dao động khoảng 2.5–2.6 GB. Ổ B còn khoảng 907 GiB.
Chưa chạy training hoặc xác nhận cấu hình dưới đây vừa VRAM.

Mục tiêu trước mắt: chạy 20 updates ở 256×256, rank=4, batch=1;
nếu loss hữu hạn và checkpoint lưu được thì tiếp tục tới 500 updates.
Đây là pilot local riêng, không tính vào 18 core runs 512×512.
Giảm số ảnh không giải quyết đáng kể VRAM khi batch size đã bằng 1.

Chạy từng khối theo thứ tự trong **Anaconda PowerShell Prompt**, cùng một cửa sổ.
Dừng ở khối đầu tiên báo lỗi. Tắt ứng dụng đang dùng GPU không cần thiết bằng tay;
không mở Gradio hoặc một phiên training khác đồng thời. Tạm tránh chế độ sleep.

## 1. Tạo môi trường riêng — chỉ làm một lần

```powershell
Set-Location -LiteralPath 'B:\Trai Phố Huế\ÚT kHờ\Year 3\Deeplearning\Project\Personalized-Text-to-Image-Generation-with-LoRA'
conda create --prefix B:\lora-local\env python=3.11 pip -y
if ($LASTEXITCODE -ne 0) { throw 'Tao environment that bai' }
conda activate B:\lora-local\env
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
if ($LASTEXITCODE -ne 0) { throw 'Cai PyTorch that bai' }
python -m pip install numpy==1.26.4 diffusers==0.32.2 transformers==4.48.3 peft==0.14.0 accelerate==1.3.0 huggingface-hub==0.28.1 tensorboard==2.18.0
if ($LASTEXITCODE -ne 0) { throw 'Cai dependencies that bai' }
python -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency conflict: dung tai day' }
```

Đây là bộ phiên bản đề xuất để kiểm tra trên Pascal, chưa phải dependency lock
đã xác nhận bằng training. Không chạy thêm `pip install -U` hoặc requirements
không pin của repo vào môi trường này. Chưa cần torchao, bitsandbytes, xformers.
CUDA 13.0 trên dòng đầu `nvidia-smi` là khả năng driver, không bắt buộc dùng wheel cu130.

## 2. Kiểm tra CUDA thật

Mỗi lần mở terminal mới: activate environment, về thư mục repo và đặt lại các biến này.

```powershell
conda activate B:\lora-local\env
$env:HF_HOME = 'B:\lora-local\hf-cache'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
$env:PIP_CACHE_DIR = 'B:\lora-local\pip-cache'
python -c "import sys,torch; print(sys.executable); print(torch.__version__,torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0),torch.cuda.get_device_capability(0)); print(torch.cuda.get_arch_list()); x=torch.randn(256,256,device='cuda',dtype=torch.float16,requires_grad=True); (x@x).float().mean().backward(); torch.cuda.synchronize(); print('CUDA forward/backward OK')"
if ($LASTEXITCODE -ne 0) { throw 'CUDA chua hoat dong; chua tai model/train' }
nvidia-smi
```

Phải có `CUDA forward/backward OK`. `torch.cuda.is_available()` riêng lẻ chưa đủ.
Nếu báo `no kernel image`, dừng và gửi log; không đổi ngẫu nhiên lên torch mới nhất.

## 3. Chuẩn bị đúng 5 ảnh và trainer

Config nguồn: `configs/local/gtx1050ti_smoke.json`.
Khối này chỉ tải mã trainer nhỏ, kiểm tra SHA256 ảnh và chép đúng subset.
Không cần tạo lại manifest hoặc chép toàn bộ data.

```powershell
@'
from pathlib import Path
import ast, csv, hashlib, json, shutil, urllib.request

cfg = json.loads(Path('configs/local/gtx1050ti_smoke.json').read_text(encoding='utf-8'))
root = Path(cfg['data_root'])
manifest = Path(cfg['manifest'])
with manifest.open(encoding='utf-8-sig', newline='') as f:
    rows = [r for r in csv.DictReader(f) if r['split'] == 'train_pool'
            and str(cfg['subset_size']) in r['subset_membership'].split(',')]
assert len(rows) == cfg['subset_size'], 'Sai so anh subset'
dst = Path(cfg['cli']['instance_data_dir'])
dst.mkdir(parents=True, exist_ok=True)
names = {Path(r['file_path']).name for r in rows}
assert len(names) == len(rows)
assert not ({p.name for p in dst.iterdir()} - names), 'Subset co file la: khong tu xoa'
for row in rows:
    src = root / row['file_path']
    assert hashlib.sha256(src.read_bytes()).hexdigest() == row['sha256'], str(src)
    shutil.copy2(src, dst / src.name)

work = Path('B:/lora-local/trainer')
work.mkdir(parents=True, exist_ok=True)
url = ('https://raw.githubusercontent.com/huggingface/diffusers/'
       + cfg['trainer_commit'] + '/examples/dreambooth/train_dreambooth_lora.py')
raw = urllib.request.urlopen(url).read()
assert hashlib.sha256(raw).hexdigest() == cfg['trainer_sha256'], 'Trainer hash changed'
(work / 'upstream.py').write_bytes(raw)
source = raw.decode('utf-8')
# Chuyen sang CPU truoc khi upcast de tranh tang VRAM luc luu final adapter.
old = 'unet = unet.to(torch.float32)'
assert source.count(old) == 1
source = source.replace(old, 'unet = unet.to(device="cpu", dtype=torch.float32)')
# Pilot khong validation/push; bo tai pipeline cuoi khong can thiet.
start = source.index('        # Final inference')
end = source.index('    accelerator.end_training()', start)
block = source[start:end]
source = (source[:start] + '        if args.validation_prompt or args.push_to_hub:\n'
          + ''.join('    ' + line if line.strip() else line
                    for line in block.splitlines(keepends=True)) + source[end:])
ast.parse(source)
(work / 'train_local.py').write_text(source, encoding='utf-8')
print('Subset OK:', dst, 'images:', len(rows))
print('Trainer ready:', work / 'train_local.py')
'@ | python -
if ($LASTEXITCODE -ne 0) { throw 'Chuan bi that bai' }
```

Hai chỉnh sửa trainer chỉ áp dụng khi kết thúc/lưu model; không thay loss hoặc
vòng training. Giữ `upstream.py` để đối chiếu. Text encoder được tính trước rồi
giải phóng, không train text encoder. Trainer này dùng alpha bằng rank.

## 4. Chạy thử 20 updates

Lần đầu lệnh này tải các thành phần SD1.5 cần thiết về ổ B; tải model chưa phải train.
Không cần `accelerate config`: chạy trực tiếp một GPU tránh cấu hình Colab cũ.

```powershell
$cfg = Get-Content 'configs/local/gtx1050ti_smoke.json' -Raw -Encoding UTF8 | ConvertFrom-Json
$out = $cfg.cli.output_dir
if (Test-Path -LiteralPath $out) { throw 'Output da ton tai: dung buoc resume, khong ghi de run cu' }
New-Item -ItemType Directory -Path $out -Force | Out-Null
Copy-Item 'configs/local/gtx1050ti_smoke.json' (Join-Path $out 'config.resolved.json')
python -m pip freeze | Out-File (Join-Path $out 'environment.txt') -Encoding utf8
Get-FileHash B:\lora-local\trainer\train_local.py | Out-File (Join-Path $out 'trainer-hash.txt')
$trainArgs = @()
foreach ($p in $cfg.cli.PSObject.Properties) {
    if ($p.Value -is [bool]) {
        if ($p.Value) { $trainArgs += "--$($p.Name)" }
    } else {
        $trainArgs += "--$($p.Name)"
        $trainArgs += [Convert]::ToString($p.Value, [Globalization.CultureInfo]::InvariantCulture)
    }
}
$log = Join-Path $out ('train-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '.log')
python -u B:\lora-local\trainer\train_local.py @trainArgs 2>&1 | Tee-Object -FilePath $log
if ($LASTEXITCODE -ne 0) { throw "Training failed: $log" }
if (-not (Test-Path (Join-Path $out 'pytorch_lora_weights.safetensors'))) { throw 'Thieu adapter' }
```

Đạt bước này khi loss không NaN/Inf, tới 20/20, có `checkpoint-20` và final adapter.
Theo dõi VRAM trong terminal thứ hai bằng `nvidia-smi -l 5`; Ctrl+C để dừng theo dõi.
Ghi tốc độ trung bình sau vài bước đầu; dự báo phần còn lại ≈ giây/update × 480.
Không dùng thời gian tải model để ước lượng tốc độ train.

## 5. Tiếp tục tới tổng 500 updates

Chỉ chạy nếu bước 4 ổn. Dùng cùng terminal để còn `$cfg`, `$out`, `$trainArgs`.
Không chạy lại bước 4 từ đầu. Mục tiêu là tổng 500, không phải thêm 500.

```powershell
if (-not (Test-Path (Join-Path $out 'checkpoint-20'))) { throw 'Chua co checkpoint-20' }
Copy-Item (Join-Path $out 'pytorch_lora_weights.safetensors') (Join-Path $out 'adapter-step20.safetensors')
$trainArgs[[Array]::IndexOf($trainArgs, '--max_train_steps') + 1] = '500'
$trainArgs[[Array]::IndexOf($trainArgs, '--checkpointing_steps') + 1] = '100'
$cfg.cli.max_train_steps = 500
$cfg.cli.checkpointing_steps = 100
$cfg | ConvertTo-Json -Depth 8 | Out-File (Join-Path $out 'config.continue-500.json') -Encoding utf8
$log = Join-Path $out ('resume-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '.log')
python -u B:\lora-local\trainer\train_local.py @trainArgs --resume_from_checkpoint latest 2>&1 | Tee-Object -FilePath $log
if ($LASTEXITCODE -ne 0) { throw "Training failed: $log" }
```

Nếu mất điện/ngắt giữa chừng, dùng lại tham số 500 và `--resume_from_checkpoint latest`.
Checkpoint chứa optimizer state để tiếp tục; final adapter riêng không thay thế checkpoint.
Không thay seed, subset, rank hoặc resolution giữa một run.

## 6. Nếu lỗi hoặc muốn chạy toàn bộ dự án

- **CUDA OOM:** giữ log, đóng ứng dụng GPU không cần thiết và thử lại bằng checkpoint
  nếu đã có. Nếu vẫn OOM ở 256/batch1, cấu hình này chưa khả thi trên máy;
  không chạy cả sweep hoặc tăng gradient accumulation để hy vọng giảm thêm VRAM.
- **Loss NaN/Inf:** dừng, giữ log. Chưa coi adapter là hợp lệ và không tiếp tục 500.
- **Lỗi dependency:** gửi traceback cùng `python -m pip freeze`; không dùng lại base Anaconda.
- **Chạy xong 500:** đây mới là adapter pilot; cần load lại, sinh ảnh, đánh giá và bổ sung
  metadata theo pipeline repo trước khi kết luận chất lượng hoặc đóng issue.
- **512×512/core matrix:** cần đo riêng cấu hình 512 và rank lớn nhất trước. Nếu không
  vừa bộ nhớ, cần GPU khác hoặc thống nhất protocol mới cho toàn bộ comparison;
  không trộn kết quả 256 với 512 trong cùng sweep.

Chưa xác nhận train thành công, chưa sửa protocol/core configs, chưa commit/push.
Lịch theo dõi Colab đã PAUSED tại thời điểm kiểm tra.

Nguồn: [PyTorch previous versions](https://docs.pytorch.org/get-started/previous-versions/),
[Diffusers trainer tại commit đã pin](https://github.com/huggingface/diffusers/blob/560fb5f4d65b8593c13e4be50a59b1fd9c2d9992/examples/dreambooth/train_dreambooth_lora.py).
