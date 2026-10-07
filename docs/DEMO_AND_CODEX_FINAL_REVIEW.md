# Demo LoRA đã train và prompt rà soát repo

## 1. Tạo ảnh demo trên máy này

Adapter pilot đã train xong 500 updates, với 5 ảnh, rank 4, 256×256. Nó không
nằm trong `artifacts/` của repo mà nằm ở `B:\lora-local`, vì vậy app Gradio hiện
tại không tự nhận adapter này. Script demo dưới đây nạp adapter trực tiếp và
lưu ảnh/metadata trong `B:\lora-local\demo`; không đưa model, adapter hay ảnh
được sinh vào Git.

Mở Anaconda PowerShell Prompt, bật đúng environment rồi vào repo:

```powershell
conda activate B:\lora-local\env
Set-Location -LiteralPath 'B:\Trai Phố Huế\ÚT kHờ\Year 3\Deeplearning\Project\Personalized-Text-to-Image-Generation-with-LoRA'
$env:HF_HOME = 'B:\lora-local\hf-cache'
$env:PYTHONIOENCODING = 'utf-8'
```

Sinh ảnh LoRA:

```powershell
python scripts/demo_local_adapter.py --seed 42 --output-dir B:\lora-local\demo
```

Sinh ảnh base để so sánh cùng prompt, seed, độ phân giải và số bước:

```powershell
python scripts/demo_local_adapter.py --base-only --seed 42 --output-dir B:\lora-local\demo
```

Ảnh lưu thành `dog_plush_lora_seed42.png` và `dog_plush_base_seed42.png`.
Mỗi ảnh có file JSON cùng tên chứa prompt, model revision, seed, hash adapter/
ảnh, phiên bản môi trường và peak VRAM. Mở ảnh bằng lệnh:

```powershell
Invoke-Item B:\lora-local\demo\dog_plush_base_seed42.png
Invoke-Item B:\lora-local\demo\dog_plush_lora_seed42.png
```

Đổi prompt bằng cách thêm `--prompt "..."`; để so sánh công bằng, dùng cùng
prompt và seed ở cả hai lệnh. Mặc định độ phân giải là 256×256 để khớp pilot đã
train; có thể thử thêm `--resolution 512` như một qualitative demo riêng, nhưng
đánh dấu rõ vì độ phân giải khác lúc train. Trainer đã cache riêng scheduler,
text encoder, tokenizer, UNet và VAE nhưng không cache file tổng
`model_index.json`; script sẽ lắp pipeline từ các component đó. Nó chạy offline,
bật CPU offload và attention slicing để giảm VRAM; thiếu component nào thì dừng,
không tải model. Safety checker không có trong cache training nên bị tắt cho demo
này và được ghi trong metadata; đừng trình bày như một kiểm tra an toàn đã bật.
Ảnh demo là qualitative example, không thay thế evaluation set hoặc metric.

## 2. Prompt Codex CLI: kiểm toán mức sẵn sàng viết báo cáo

Chạy Codex CLI từ thư mục gốc repo rồi dán prompt này. Prompt chỉ yêu cầu kiểm
toán và báo cáo phát hiện; chưa cho phép sửa code, xóa file hay tạo kết luận thiếu
bằng chứng.

```text
Đóng vai reviewer kỹ thuật và research QA cho đồ án. Đọc AGENTS.md, toàn bộ
docs/planning/01_PROJECT_BRIEF.md đến 06_ISSUES_AND_MILESTONES.md, protocol,
decision log, README, runbook, DATA docs, verification docs và mã nguồn liên quan.
Đọc yêu cầu thi tại D:/AI thực chiến/Lecture Vinuni/4_Exam Requirements.pdf.

Mục tiêu: xác định repo đã đủ bằng chứng để viết báo cáo cuối cùng và trình bày
hay mới đủ viết báo cáo tiến độ. Đối chiếu từng yêu cầu trong PDF: 10–15 trang
(không tính tài liệu tham khảo/phụ lục), abstract 150–200 từ, introduction/RQ,
related work, dataset/preprocessing và URL/phiên bản, baseline, phương pháp,
ba experimental setups, kết quả có diễn giải, lỗi/qualitative analysis, kết luận
và limitations, 5–10 tài liệu tham khảo, contribution table; thêm README tái lập
được kết quả và bốn slide/3 phút.

Bằng chứng local cần kiểm tra trực tiếp:
- Run dog_plush_n5_r4_256_ts42 dùng 5 ảnh, rank 4, 256x256, seed 42, tổng 500
  updates. Adapter ở B:/lora-local/runs/dog_plush_n5_r4_256_ts42/
  pytorch_lora_weights.safetensors; log resume ở cùng thư mục. Đây là pilot local
  riêng, không phải protocol v1 ở 512x512.
- Ảnh demo (nếu đã sinh) ở B:/lora-local/demo; đọc metadata JSON cạnh ảnh.
- Không giả định adapter đã được load/inference chỉ vì file safetensors tồn tại.
  Phân biệt training complete, adapter-load verified, qualitative demo, và
  evaluation complete.
- Đọc results CSV và run artifacts thực tế; phân biệt header-only, placeholder,
  planned/dry-run với số liệu experiment thật. Xác nhận trạng thái clean/dirty
  Git và không ghi đè các thay đổi đang có của người khác.

Không chạy training, inference, tải model/data, cài package, tests đắt, hoặc
thay đổi/xóa file. Không bịa tên thành viên, ratings, URLs, kết quả, citations,
runtime hay lỗi. Nếu dữ kiện thiếu, ghi rõ và dùng [CẦN NHÓM XÁC NHẬN] như một
placeholder có nhãn trong checklist, không điền suy đoán.

Trả kết quả bằng tiếng Việt, súc tích nhưng đủ hành động:
1. Kết luận một trong: viết được báo cáo kết quả; viết được báo cáo có giới hạn;
   chưa thể viết trung thực phần kết quả.
2. Ma trận PDF requirement -> bằng chứng/path -> PASS/PARTIAL/MISSING.
3. Các mâu thuẫn docs/code và tài liệu cũ cần cập nhật trước nộp.
4. Danh sách việc P0 theo thứ tự, mỗi việc có artifact cần tạo và tiêu chí xong.
5. Run IDs/kết quả được phép nêu và những kết luận chưa được phép nêu.
6. Dàn ý 10–15 trang và outline đúng 4 slides, dùng riêng cho trạng thái bằng
   chứng hiện có; không viết hộ kết quả chưa chạy.
Kết thúc bằng danh sách ngắn thông tin chỉ nhóm/người dùng cung cấp được.
```

## 3. Prompt Codex CLI: rà file trước khi public repo

Prompt này yêu cầu inventory và kế hoạch. Repo instructions cấm xóa/move file
legacy khi chưa được duyệt; do đó Codex phải dừng trước thao tác xóa và xin duyệt
danh sách đường dẫn cụ thể. Không dùng `git clean -fdx`.

```text
Rà soát repo để chuẩn bị public GitHub cho đồ án/báo cáo. Trước tiên đọc
AGENTS.md và docs/planning/01_PROJECT_BRIEF.md đến 06_ISSUES_AND_MILESTONES.md.
Kiểm tra git status, git diff, git ls-files, file ignored/untracked, README,
DATA.md/data docs, manifests, LICENSES.md, configs, prompt bank, scripts, tests,
verification records và results.

Pha này CHỈ kiểm toán, không sửa và không xóa gì. Không commit/push, không chạy
git clean, không force operation, không xóa/move thư mục, không xóa legacy files.
Không đọc/đưa vào Git raw/private photos, held-out images, model cache, adapter,
checkpoint, generated images, logs có dữ liệu nhạy cảm, credentials hoặc env.

Phân loại từng candidate:
A. Phải giữ để source/reproduction/report.
B. Chỉ là local scratch/cache/test output, có thể xóa sau khi duyệt.
C. Chưa rõ nguồn gốc hoặc có thể cần cho reproduction; giữ nguyên.
D. Private/large artifact; giữ ngoài Git và xác nhận .gitignore.

Với mỗi candidate ghi exact path, tracked/untracked/ignored, kích thước, lý do,
references từ code/docs, hậu quả nếu bỏ, phân loại, và hành động đề xuất.
Kiểm tra cả file bị ignore bằng git check-ignore -v; đừng suy ra rằng mọi ignored
file đều rác (data cá nhân/model cache có thể nằm ở đó). Dùng dry-run/listing,
không xóa file để “thử”.

Đặc biệt xác nhận:
- README phản ánh code/commands hiện tại, không còn tuyên bố placeholder lỗi thời.
- DATA.md có dataset source/URL hoặc giải thích rõ custom self-captured dataset,
  version, split, preprocessing, script, license/provenance và link tải phù hợp.
- Không stage dữ liệu riêng trong data/raw, data/processed, data/eval_refs; ảnh
  demo và model artifacts không nằm trong Git.
- Local LoRA pilot 256x256/rank4 được ghi đúng là pilot, không trộn vào core v1.
- Secrets hoặc local absolute paths không lọt vào tracked files.
- Giữ lại manifests, prompt/seed bank, code, tests, configs, docs, attribution,
  experiment logs đã scrub thông tin riêng nhưng cần cho bằng chứng.

Xuất báo cáo gồm: tổng quan git status; rủi ro public; bảng file giữ/xóa/cần
quyết định; đề xuất .gitignore; thiếu sót README/DATA.md/license; và các lệnh
preview an toàn. KHÔNG chạy lệnh xóa. Dừng và yêu cầu người dùng duyệt chính xác
các path thuộc nhóm B trước khi xóa. Chỉ sau khi họ chấp thuận từng path mới được
xóa đúng các path đó; giữ nhóm C/D và tuyệt đối không dùng glob hoặc recursive
delete. Sau khi được duyệt, cập nhật checklist và xác nhận bằng git status.
```

Sau khi duyệt repo, dùng review cuối để đảm bảo tracked diff sạch; sau đó chủ
repo tự quyết định commit/push. Không public raw data, adapter hoặc ảnh đầu ra
nếu chưa có quyền chia sẻ rõ ràng.
