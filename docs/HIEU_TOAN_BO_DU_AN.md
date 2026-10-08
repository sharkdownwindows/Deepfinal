# Hiểu toàn bộ dự án LoRA: từ vấn đề, issues đến code và bước đang làm

Ngày đối chiếu: **2026-10-08**. Source snapshot: `dd0564444699205e2568810f141491f1bfd8f747`.

Tài liệu dành cho người cần hiểu để tự chạy, debug và bảo vệ đồ án; không phải báo cáo kết quả đã hoàn tất. Các ví dụ số học được ghi rõ là minh họa, không phải metric thực nghiệm.

Đã đối chiếu planning 01–06, code, tests, verification notes, bốn bảng kết quả và trạng thái issues trên GitHub. Không chạy training/inference/scoring; không mở ảnh riêng, weights, checkpoint hay log riêng để viết tài liệu này. Thông tin adapter 500 bước dựa trên xác nhận của người dùng và tài liệu demo hiện có, không phải một lần kiểm chứng lại weight trong phiên này.

## 1. Kết luận trước: dự án làm gì và bạn đang ở đâu?

Dự án nghiên cứu cách dạy Stable Diffusion v1.5 sinh **một vật thể cụ thể của nhóm**, từ ít ảnh, bằng DreamBooth-LoRA. Sau đó thay đổi số ảnh hoặc LoRA rank để xem chất lượng thay đổi thế nào.

Ví dụ: model có thể biết “plush toy”, nhưng chưa biết đúng con thú bông `dog_plush` của nhóm. Ta muốn nó sinh đúng con đó trên bàn, ở bãi biển hoặc từ góc nhìn khác, thay vì một thú bông bất kỳ.

Cần giải quyết đồng thời hai bài toán:

1. **Bài toán mô hình:** học đặc điểm riêng của vật thể nhưng vẫn làm theo prompt mới.
2. **Bài toán thực nghiệm:** chứng minh sự thay đổi bằng so sánh công bằng, truy vết được và tái lập được; không chỉ chọn vài ảnh đẹp.

Trạng thái đến bước bạn đang làm:

```text
Protocol, manifests, prompts và phần lớn code đã có
                         |
                         v
Local pilot: dog_plush, 5 ảnh, rank 4, 256×256, 500 updates
                         |
                         v
Bạn đang nối adapter vào inference/evaluation
  Lỗi đã gửi: package import + cú pháp PowerShell
                         |
                         v
Chưa có bằng chứng batch đánh giá + bảng điểm thật trong results/
                         |
                         v
Còn core sweeps, human evaluation, analysis và báo cáo kết quả
```

**500 training steps không đồng nghĩa hoàn thành dự án.** Nó là một mốc của một run. Còn phải load đúng adapter, sinh đúng batch, chấm đúng, kiểm tra độ đầy đủ và so sánh với các điều kiện khác.

Nguồn: [project brief](planning/01_PROJECT_BRIEF.md), [hướng dẫn demo local](DEMO_AND_CODEX_FINAL_REVIEW.md), [QA-01 verification](verification/QA-01.md).

## 2. Những khái niệm phải phân biệt

### 2.1 Stable Diffusion, DreamBooth và LoRA không phải ba model thay thế nhau

| Thành phần | Vai trò trong dự án |
|---|---|
| Stable Diffusion v1.5 | Backbone pretrained: đã có khả năng text-to-image tổng quát |
| DreamBooth | Cách cá nhân hóa theo một subject từ vài ảnh và một định danh trong prompt |
| LoRA | Cách biểu diễn phần trọng số cần học bằng các ma trận nhỏ, thay vì fine-tune toàn bộ model |
| Diffusers | Thư viện và trainer/pipeline được dùng để thực hiện training và inference |
| DINOv2, CLIP | Model chấm các khía cạnh của ảnh sinh; không phải adapter được train trong đồ án |

DreamBooth gắn định danh với subject qua một số ảnh, để tái hiện subject ở ngữ cảnh mới. Dự án dùng ý tưởng này với LoRA và cấu hình riêng: text encoder frozen, prior preservation tắt trong core protocol; không nên nói đã tái hiện đầy đủ mọi thiết lập của bài báo gốc. Nguồn khái niệm: [DreamBooth](https://arxiv.org/abs/2208.12242).

### 2.2 LoRA học cái gì?

Với một lớp tuyến tính có trọng số `W`, LoRA giữ `W` cố định và học phần bổ sung:

```text
W_effective = W + (alpha / rank) × B × A
```

Nếu `W` có kích thước `d_out × d_in`, thì `A` có kích thước `rank × d_in`, `B` có kích thước `d_out × rank`. Số tham số bổ sung là `rank × (d_in + d_out)`.

Ví dụ số học tự dựng: `W` là `1000 × 1000`, tương ứng 1.000.000 tham số. Với rank 4, hai ma trận LoRA có tổng 8.000 tham số. Đây là minh họa cho một lớp, **không phải số tham số đã đo của adapter dự án**.

Rank lớn tăng khả năng biểu diễn của phần cập nhật, nhưng không bảo đảm ảnh đẹp hơn. Có thể học thêm đặc điểm subject, cũng có thể học quá chặt background/pose. Core giữ `alpha = rank` để không đồng thời thay đổi hệ số `alpha/rank` khi nghiên cứu rank. Nguồn nguyên lý: [LoRA](https://arxiv.org/abs/2106.09685); cấu hình áp dụng trong dự án: [technical spec](planning/03_PROJECT_OVERVIEW_AND_TECHNICAL_SPEC.md).

Trong repo, xem `LoraConfig`, `requires_grad_` và `target_modules` ở [trainer Diffusers được lưu trong repo](../examples/dreambooth/train_dreambooth_lora.py). Đây là mã upstream/vendored, không phải thuật toán LoRA do nhóm phát minh. Trainer local đã dùng có thể là một bản pin/patch khác; phải đối chiếu revision/hash trainer của run, không mặc định file trong repo chính là file đã train weight của bạn.

### 2.3 Training khác inference như thế nào?

Training đi theo ý tưởng:

```text
Ảnh train -> VAE encode thành latent -> thêm noise tại timestep ngẫu nhiên
Prompt -> tokenizer/text encoder -> conditioning
Latent nhiễu + timestep + conditioning -> UNet có LoRA -> dự đoán target
Loss giữa prediction và target -> backward -> cập nhật tham số LoRA
```

Trong trainer, tìm `noise_scheduler.add_noise`, `prediction_type`, `F.mse_loss`, `optimizer.step`, `global_step`. Target có thể là noise hoặc velocity tùy scheduler config; không suy diễn mọi nhánh đều cùng một target.

Inference bắt đầu từ noise theo seed, chạy các bước khử nhiễu có conditioning của prompt rồi dùng VAE decode ra ảnh. Lúc này dùng trọng số đã học, không gọi optimizer để học tiếp.

| Đại lượng | Ý nghĩa | Ví dụ trong dự án |
|---|---|---|
| Training step/update | Một lần cập nhật optimizer | Local pilot đã được báo train tới 500 |
| Epoch | Một lượt qua tập train; không đồng nghĩa một update | Phụ thuộc số ảnh, batch và accumulation |
| Inference step | Một bước sampling/khử nhiễu khi sinh ảnh | Core 30; demo local mặc định 25 |
| Training seed | Điều khiển randomness của quá trình train | Core dùng 42 |
| Generation seed | Điều khiển randomness khi sinh ảnh | Evaluation dùng 11, 22, 33, 44 |
| Rank | Kích thước hạng thấp của cập nhật LoRA | Core 4, 16, 32 |
| LoRA scale | Mức tác động adapter lúc inference | Core mặc định 1.0 |

Một seed không xác định kết quả nếu model, prompt, scheduler, resolution hoặc môi trường thay đổi. Cùng seed cũng không bảo đảm bit-identical giữa các GPU/library versions.

### 2.4 Base weights, adapter và checkpoint

- **Base weights:** trọng số SD1.5 pretrained, chứa năng lực sinh ảnh tổng quát.
- **Adapter:** phần cập nhật LoRA đã học. File `pytorch_lora_weights.safetensors` không phải toàn bộ Stable Diffusion; phải nạp cùng backbone tương thích.
- **Training checkpoint:** phục vụ resume, gồm trạng thái training cần thiết tùy trainer, chẳng hạn optimizer/scheduler. Chỉ có final adapter không đủ để khôi phục nguyên trạng optimizer.
- **Generated image:** đầu ra inference; không chứng minh model đã được đánh giá đầy đủ.
- **Metadata:** hồ sơ liên kết ảnh với run, prompt, seed, model, adapter và settings.

Tách các loại artifact này giúp hiểu vì sao “đã có weight” vẫn có thể lỗi import, thiếu model cache hoặc thiếu config khi chạy evaluation.

## 3. Hai research questions và vì sao chỉ có 18 core runs

### RQ1: Thay số ảnh, giữ rank

Với mỗi concept: `n = 1, 3, 5, 10`, giữ `rank = 16`, cùng ngân sách 500 updates và các biến kiểm soát khác. Tổng: `3 concepts × 4 = 12 runs`.

### RQ2: Thay rank, giữ số ảnh

Với mỗi concept: `rank = 4, 16, 32`, giữ `n = 5`. Có 9 ô so sánh, nhưng ba ô `n5/r16` đã thuộc RQ1. Vì vậy chỉ cần thêm 6 runs.

| n / rank | r4 | r16 | r32 |
|---|---|---|---|
| n1 | Không thuộc core | RQ1 | Không thuộc core |
| n3 | Không thuộc core | RQ1 | Không thuộc core |
| n5 | RQ2 | Chung RQ1/RQ2 | RQ2 |
| n10 | Không thuộc core | RQ1 | Không thuộc core |

Tổng `3 × (4 + 3 − 1) = 18`, không phải full factorial `3 × 4 × 3 = 36`.

Nested subsets nghĩa là `D1 ⊂ D3 ⊂ D5 ⊂ D10`: ảnh trong subset nhỏ phải nằm trong subset lớn. Nếu n1 và n3 lấy ảnh tùy ý khác hẳn nhau, kết quả vừa đổi số ảnh vừa đổi cách chọn ảnh, khó diễn giải.

Ngay cả nested subsets vẫn có giới hạn: giữ 500 updates khiến subset nhỏ được lặp nhiều hơn. Với batch 1 và accumulation 1, minh họa gần đúng 500 lượt chọn ảnh tương ứng trung bình 500 lượt/ảnh ở n1, 100 lượt/ảnh ở n5, 50 lượt/ảnh ở n10. Vì vậy RQ1 là **ảnh hưởng của số ảnh dưới ngân sách update cố định**, không phải thí nghiệm đã loại mọi confound.

Một training seed không đo được training-seed variance. Bốn generation seeds không thay thế bốn lần train độc lập.

Nguồn thực thi: [core invariants](../src/personalized_t2i/config.py), [ma trận RQ1](../configs/ml04_sweep.yaml), [experiment protocol](protocol.md).

## 4. Issues là chuỗi bằng chứng, không chỉ danh sách file cần viết

Một issue gồm vấn đề, dependencies, acceptance criteria và artifact cần giao. Milestone là cổng kiểm tra một nhóm issue; không phải cứ đủ số file là qua cổng.

Snapshot GitHub ngày 2026-10-08: **#1 CLOSED, #2–#24 OPEN**. Trạng thái này được đọc trực tiếp bằng GitHub CLI. OPEN không đồng nghĩa chưa có code; CLOSED cũng không thay cho kiểm tra bằng chứng cụ thể. Bảng sau giải thích vấn đề và liên kết implementation, không tự đóng hoặc chứng nhận hoàn thành issue.

### 4.1 Khóa đề bài và dữ liệu

| Issue | Vấn đề cần giải quyết | Code/docs cần đọc | Bằng chứng cần có |
|---|---|---|---|
| [#1 PROD-01](https://github.com/sharkdownwindows/Personalized-Text-to-Image-Generation-with-LoRA/issues/1) | Ngăn đổi mục tiêu và thí nghiệm tùy hứng | [protocol](protocol.md), [decision log](decision_log.md), planning 01–06 | RQ, matrix, biến kiểm soát và giới hạn claim đã khóa; issue đã CLOSED |
| [#2 OPS-01](https://github.com/sharkdownwindows/Personalized-Text-to-Image-Generation-with-LoRA/issues/2) | Người khác biết cần môi trường nào, lưu output ra sao | [pyproject](../pyproject.toml), [requirements](../requirements.in), [environment.py](../src/personalized_t2i/environment.py), [.gitignore](../.gitignore) | Môi trường thực tế và provenance; dependency input không tự động là lock đã kiểm chứng |
| [#3 DATA-01](https://github.com/sharkdownwindows/Personalized-Text-to-Image-Generation-with-LoRA/issues/3) | Chọn subject có định danh, nguồn và quyền sử dụng | [concept registry](../data/manifests/concepts.csv), [DATA-01 note](verification/DATA-01.md) | Ba concept, token, class noun, ownership/provenance; không suy quyền public từ chữ self-captured |
| [#4 DATA-02](https://github.com/sharkdownwindows/Personalized-Text-to-Image-Generation-with-LoRA/issues/4) | Ngăn sai ảnh, thiếu ảnh và train/eval leakage | [manifest docs](../data/manifests/README.md), [validate.py](../src/personalized_t2i/data/validate.py), [CLI](../scripts/validate_data.py) | Mỗi concept 10 train/3 held-out, hash/split hợp lệ |
| [#5 DATA-03](https://github.com/sharkdownwindows/Personalized-Text-to-Image-Generation-with-LoRA/issues/5) | Chọn subset và preprocessing nhất quán | [prepare.py](../src/personalized_t2i/data/prepare.py), [preprocessing docs](data_03_preprocessing.md) | Nested membership, quy tắc selection, version và preprocessing provenance |
| [#6 DATA-04](https://github.com/sharkdownwindows/Personalized-Text-to-Image-Generation-with-LoRA/issues/6) | Không đổi prompt/seed để chọn ảnh đẹp | [prompt bank](../prompt_bank/evaluation_prompts.yaml), [seed bank](../prompt_bank/generation_seeds.yaml) | 8 prompts/concept, 4 seeds và version cố định |

### 4.2 Làm một đường chạy end-to-end

| Issue | Vấn đề cần giải quyết | Code/docs cần đọc | Phân biệt implementation với hoàn tất |
|---|---|---|---|
| [#7 ML-01](https://github.com/sharkdownwindows/Personalized-Text-to-Image-Generation-with-LoRA/issues/7) | GPU/backbone có thực sự chạy được không? | [runner](../src/personalized_t2i/training/runner.py), [local training guide](LOCAL_TRAIN_GTX1050TI.md) | Pilot cần train/load/inference và đo tài nguyên; local 256 không tự chứng minh core 512 khả thi |
| [#8 BE-01](https://github.com/sharkdownwindows/Personalized-Text-to-Image-Generation-with-LoRA/issues/8) | Chặn cấu hình sai, tên run sai | [config.py](../src/personalized_t2i/config.py), [preflight.py](../src/personalized_t2i/preflight.py), [test_config](../tests/test_config.py) | Có logic/tests; phải kiểm tra validator nào thực sự được entry point gọi |
| [#9 ML-02](https://github.com/sharkdownwindows/Personalized-Text-to-Image-Generation-with-LoRA/issues/9) | Biến YAML thành training và lưu kết quả | [train_run.py](../scripts/train_run.py) → [run_training](../src/personalized_t2i/training/runner.py) | Code wrapper đã có; cần run thật và adapter reload, không chỉ exit code |
| [#10 BE-02](https://github.com/sharkdownwindows/Personalized-Text-to-Image-Generation-with-LoRA/issues/10) | Không ghi đè run; biết run thành công/lỗi và dùng gì | [registry.py](../src/personalized_t2i/registry.py), [test_registry](../tests/test_registry.py) | Lifecycle và provenance đã có code; lỗi phải được giữ lại |
| [#11 ML-03](https://github.com/sharkdownwindows/Personalized-Text-to-Image-Generation-with-LoRA/issues/11) | Sinh batch cùng prompt/seed, đủ mẫu, phân biệt baseline | [generate CLI](../scripts/generate_eval_set.py) → [generate.py](../src/personalized_t2i/inference/generate.py) | Có implementation/fake-pipeline tests; cần batch thật đầy đủ |
| [#12 EVAL-01](https://github.com/sharkdownwindows/Personalized-Text-to-Image-Generation-with-LoRA/issues/12) | Đo mức giống subject | [fidelity.py](../src/personalized_t2i/evaluation/fidelity.py), [test_fidelity](../tests/test_fidelity.py) | DINO held-out-centroid scorer có code; tests fake không phải điểm model thật |
| [#13 EVAL-02](https://github.com/sharkdownwindows/Personalized-Text-to-Image-Generation-with-LoRA/issues/13) | Đo mức làm theo prompt | [alignment.py](../src/personalized_t2i/evaluation/alignment.py), [test_alignment](../tests/test_alignment.py) | CLIP scorer/normalization có code; cần ảnh thật và encoder thật |
| [#14 QA-01](https://github.com/sharkdownwindows/Personalized-Text-to-Image-Generation-with-LoRA/issues/14) | Các khâu có nối được với nhau không? | [vertical slice CLI](../scripts/run_vertical_slice.py), [verification](verification/QA-01.md) | Script có đường train→load→generate→score; note vẫn chưa xác nhận một lượt thật hoàn chỉnh |

### 4.3 Chạy nghiên cứu, không chỉ chạy model

| Issue | Vấn đề cần giải quyết | Code/docs cần đọc | Phần còn phải có |
|---|---|---|---|
| [#15 ML-04](https://github.com/sharkdownwindows/Personalized-Text-to-Image-Generation-with-LoRA/issues/15) | Thu dữ liệu trả lời RQ1 | [run_sweep.py](../scripts/run_sweep.py), [matrix](../configs/ml04_sweep.yaml) | Có runner cho 12 ô; cần 12 run thật và failed/missing records |
| [#16 ML-05](https://github.com/sharkdownwindows/Personalized-Text-to-Image-Generation-with-LoRA/issues/16) | Thu dữ liệu trả lời RQ2 | [run_ml05_sweep.py](../scripts/run_ml05_sweep.py) | Script hiện chỉ dog_plush và còn contract/path lỗi; chưa phải sweep 3-concept hoàn chỉnh |
| [#17 EVAL-03](https://github.com/sharkdownwindows/Personalized-Text-to-Image-Generation-with-LoRA/issues/17) | Chuyển điểm từng ảnh thành bảng so sánh đúng | [evaluate CLI](../scripts/evaluate_outputs.py), [aggregate.py](../src/personalized_t2i/evaluation/aggregate.py) | Cần sửa grouping base/adapter và kiểm tra validity/completeness trước dùng aggregate |
| [#18 EVAL-04](https://github.com/sharkdownwindows/Personalized-Text-to-Image-Generation-with-LoRA/issues/18) | Kiểm tra automated metrics bằng đánh giá người | [prepare_human_eval.py](../scripts/prepare_human_eval.py), [human protocol](../configs/human_eval_v1.yaml), [guide](human_evaluation.md) | Tooling không thay thế 60 ảnh thật và ít nhất 5 raters |
| [#19 EVAL-05](https://github.com/sharkdownwindows/Personalized-Text-to-Image-Generation-with-LoRA/issues/19) | Giải thích model thất bại kiểu gì | [taxonomy](failure_taxonomy.md), [failures.py](../src/personalized_t2i/evaluation/failures.py), [validator CLI](../scripts/validate_failure_cases.py) | Cần cases thật có run/prompt/seed và base comparison; bảng hiện chưa có cases |

### 4.4 Chuyển bằng chứng thành sản phẩm nộp

| Issue | Vấn đề cần giải quyết | Code/docs cần đọc | Phần còn phải có |
|---|---|---|---|
| [#20 FE-01](https://github.com/sharkdownwindows/Personalized-Text-to-Image-Generation-with-LoRA/issues/20) | Duyệt, đối chiếu output và provenance | [app/demo.py](../app/demo.py) | Explorer đã có code; cần kiểm tra trên artifacts thật, không coi UI là trainer |
| [#21 REP-01](https://github.com/sharkdownwindows/Personalized-Text-to-Image-Generation-with-LoRA/issues/21) | Biểu đồ trả lời RQ, grid so sánh công bằng | [charts.py](../src/personalized_t2i/reporting/charts.py), [grids.py](../src/personalized_t2i/reporting/grids.py), [report CLI](../scripts/build_report_assets.py) | Libraries có code nhưng CLI còn “not implemented”; cần tích hợp và nguồn số liệu thật |
| [#22 DOC-01](https://github.com/sharkdownwindows/Personalized-Text-to-Image-Generation-with-LoRA/issues/22) | Viết kết luận đúng với dữ liệu | [report outline](report_outline.md), [report handoff](CODEX_CLI_OVERLEAF_REPORT_PROMPT_2026-10-07.md) | Methods có thể viết trước; findings phải có evidence, kể cả null/negative results |
| [#23 QA-02](https://github.com/sharkdownwindows/Personalized-Text-to-Image-Generation-with-LoRA/issues/23) | Người khác có làm lại được không? | [runbook](runbook.md), [README](../README.md), artifact/environment contracts | Một người không viết trainer chạy representative run và ghi khác biệt |
| [#24 DOC-02](https://github.com/sharkdownwindows/Personalized-Text-to-Image-Generation-with-LoRA/issues/24) | Trình bày ổn định khi live demo có rủi ro | [demo guide](DEMO_AND_CODEX_FINAL_REVIEW.md), [roadmap](planning/04_TEAM_WORKFLOW_AND_ROADMAP.md) | Slides, backup demo và rehearsal thật; một script tồn tại chưa chứng minh đủ |

Milestone đọc theo thứ tự: **M0 scope → M1 vertical slice → M2 data/config freeze → M3 core runs → M4 evidence → M5 release candidate → M6 submission**. Không nên tính “% hoàn thành” bằng số issue đóng hoặc số dòng code.

## 5. Đi theo một mẫu dữ liệu từ đầu đến cuối

### 5.1 Dataset không phải một thư mục ảnh tùy ý

Ba concept thật trong [concept registry](../data/manifests/concepts.csv):

| Concept ID | Trigger token | Class noun |
|---|---|---|
| cat_mug | zzobj01 | mug |
| dog_plush | zzobj02 | plush toy |
| blue_white_vase | zzobj03 | vase |

Manifest có các thông tin quan trọng: `file_path`, `sha256`, `concept_id`, `split`, `subset_membership`, `caption`, nguồn/quyền sử dụng. Nó giống “danh sách mẫu đã khóa” để mọi người dùng cùng tập ảnh.

Ví dụ diễn giải membership, không phải một ảnh mới được thêm vào data:

```text
split=train_pool, subset_membership="1,3,5,10"
=> ảnh này được dùng trong cả bốn subset.

split=train_pool, subset_membership="5,10"
=> không dùng ở n1/n3; có dùng ở n5/n10.

split=heldout
=> chỉ làm reference đánh giá, không đưa vào training subset.
```

[DATA.md](../DATA.md) mô tả dataset tự chụp v1, 30 train + 9 held-out, restricted download link và archive hash. Không có URL của một public benchmark vì đây không phải dataset public bên thứ ba. Tài liệu này không tải lại archive hay xác nhận quyền truy cập Drive.

`prepare.py` có EXIF correction, RGB conversion, kiểm tra ảnh nguồn vuông, Lanczos resize 512, JPEG quality 95, không random flip. Phân biệt **bản processed 512** với **ảnh trainer thực sự đọc**: runner hiện chọn ảnh raw theo manifest rồi trainer áp dụng transform training. Có processed folder không tự bảo đảm trainer sử dụng folder đó.

### 5.2 Config và run ID khóa “đã chạy cái gì”

Ví dụ tên `dog_plush_n5_r16_ts42`:

- Subject `dog_plush`.
- 5 ảnh theo membership trong manifest.
- Rank 16.
- Training seed 42.

Tên là nhãn, không thay thế config. Cần kiểm tra `model`, `data`, `training`, `inference`, `output` trong resolved config. Đặc biệt không gắn nhãn core chỉ vì run ID giống convention.

Code hiện có hai lớp validation cần phân biệt:

- [config.py](../src/personalized_t2i/config.py): core protocol invariants.
- `validate_training_config()` trong [runner.py](../src/personalized_t2i/training/runner.py): validation của đường training đang được CLI gọi; có xử lý pilot.

Không phải mọi entry point tự động gọi cùng một validator. Đọc call chain thay vì suy từ tên file `schema.yaml` rằng toàn bộ pipeline đã được enforce.

### 5.3 Training runner làm những việc nào?

Đọc [train_run.py](../scripts/train_run.py) trước, rồi tìm các hàm này trong [runner.py](../src/personalized_t2i/training/runner.py):

1. `validate_training_config`: kiểm tra config/output và điều kiện training.
2. `resolve_training_subset`: lấy đúng membership, count và hash của ảnh được chọn.
3. `materialize_training_subset`: chép subset vào thư mục training data của run; không đưa held-out vào train.
4. `build_trainer_command`: chuyển config thành tham số cho official Diffusers trainer.
5. `run_training`: gọi trainer, thu log, lấy adapter, load lại và sinh một ảnh pilot.
6. `write_pilot_metadata`: lưu metadata của ảnh pilot riêng với batch evaluation.

Runner quản lý thực thi; thuật toán tối ưu nằm trong trainer Diffusers được truyền qua `--diffusers-script`. Có wrapper không có nghĩa nhóm đã tự viết toàn bộ Stable Diffusion.

`registry.create_run()` lưu resolved config, environment và provenance trước khi chạy. Lifecycle là `queued → running → completed/failed`. Ở runner hiện tại, `completed` có nghĩa phần training và reload/sinh ảnh pilot đã thành công; **chưa đồng nghĩa batch 8×4 và DINO/CLIP đều hoàn tất**.

Layout điển hình do runner/batch tạo, không phải khẳng định local adapter của bạn đã có đủ những file này:

```text
artifacts/RUN_ID/
  config.resolved.yaml       cấu hình snapshot của run
  environment.json          Git/Python/packages/GPU/CUDA thông qua collector
  provenance.json           model + dataset + manifest hash + training settings
  status.json               lifecycle/error
  training_subset.json      ảnh nào thực sự được chọn
  training_data/            bản subset riêng cho trainer
  logs/train.log            log trainer, giữ private khi chưa scrub
  trainer_output/           output gốc, gồm checkpoint theo trainer
  adapter/                  final adapter được runner sao chép
  generations/adapter_pilot.png
  pilot_metadata.jsonl      một ảnh kiểm tra reload
  metrics.json              timing/GPU-memory metadata của runner
  generations/evaluation/base/      batch chỉ xuất hiện sau bước generate
  generations/evaluation/adapter/
  metadata.jsonl            hồ sơ batch evaluation
```

Registry cũng tạo `checkpoints/`, nhưng runner truyền output cho trainer vào `trainer_output/`; không mặc định một thư mục có tên `checkpoints/` là nơi duy nhất chứa checkpoint thực tế.

### 5.4 Batch generation: từ một adapter thành tập đánh giá

[generate_eval_set.py](../scripts/generate_eval_set.py) gọi `generate_evaluation_batch()` trong [generate.py](../src/personalized_t2i/inference/generate.py).

Luồng chính:

1. Đọc resolved config, model ID/revision và concept ID.
2. Đọc 8 prompts của concept và 4 seeds từ bank.
3. Nạp base model; nếu truyền adapter thì chạy hai modes `base` và `adapter`.
4. Tạo RNG mới cho từng cặp prompt/seed; giữ các tham số inference cố định.
5. Lưu ảnh và record có `generation_mode`, prompt, seed, revision, resolution, steps, scale.
6. Kiểm tra coverage; ghi `metadata.jsonl` sau khi batch hoàn tất.

Số mẫu: `8 × 4 = 32` cho mỗi mode. Khi truyền adapter, current generator tạo **32 base + 32 adapter = 64 ảnh/run**. Không truyền adapter thì chỉ có 32 ảnh base; đó không phải đánh giá LoRA.

Current implementation dùng cùng prompt-bank text cho base và adapter. Không tự mô tả baseline là class-only prompt nếu chưa đổi/version protocol tương ứng. CLIP normalization là bước tính metric riêng, không phải bằng chứng prompt generation đã được sửa thành class-only.

Nếu đã có metadata/output cùng tên, generator từ chối ghi đè. Batch lỗi giữa chừng có thể để lại ảnh nhưng chưa có final metadata; đó là partial output cần giữ và xử lý có kiểm soát, không được xóa để làm như chưa từng thất bại.

### 5.5 Chấm DINO: “Có giống đúng vật thể không?”

[Dinov2FidelityScorer](../src/personalized_t2i/evaluation/fidelity.py) nạp frozen `facebook/dinov2-base`, encode ba held-out references, lấy trung bình embedding rồi so cosine với embedding ảnh sinh:

```text
h_bar = (phi(h1) + phi(h2) + phi(h3)) / 3
DINO(g) = cosine(phi(g), h_bar)
cosine(a,b) = dot(a,b) / (norm(a) × norm(b))
```

Mô tả trên bám current code: lấy mean của embeddings trước, cosine similarity thực hiện chuẩn hóa khi so sánh. Không nói code đã normalize từng reference trước khi tính centroid nếu chưa có bước đó.

Ví dụ hình học tự dựng: hai vector `[1,0]` và `[1,0]` cho cosine 1; `[1,0]` và `[0,1]` cho 0. Điểm này không phải “xác suất đúng subject” hoặc “accuracy %”. Background/layout có thể ảnh hưởng DINO toàn ảnh, nên vẫn cần human review/failure analysis.

Held-out dùng để tránh đo chủ yếu mức khớp với ảnh đã train. Nhưng chỉ ba reference cũng không phải một ground truth identity metric hoàn hảo.

### 5.6 Chấm CLIP: “Có làm theo prompt không?”

[ClipPromptScorer](../src/personalized_t2i/evaluation/alignment.py) encode ảnh và text bằng frozen `openai/clip-vit-base-patch32`, rồi tính cosine.

`normalize_prompt_for_clip()` biến:

```text
a photo of zzobj02 plush toy on a wooden table
-> a photo of plush toy on a wooden table
```

Lý do: CLIP pretrained không được train riêng để hiểu trigger token của nhóm; scorer cần đo sự khớp với class/context. Tránh thay token thành class noun rồi để lặp “plush toy plush toy”.

Hai metric trả lời khác nhau:

- Đúng thú bông nhưng sai bối cảnh: có thể fidelity tốt, prompt adherence kém.
- Đúng bãi biển nhưng sai con thú bông: có thể CLIP khá, identity chưa đạt.
- Không kết luận từ một metric đơn lẻ; không cộng tùy tiện DINO và CLIP thành một “điểm tổng”.

### 5.7 Per-sample → aggregate → figure

[evaluate_outputs.py](../scripts/evaluate_outputs.py) gọi scorer và ghi per-sample CSV. DINO/CLIP upsert nhằm bổ sung metric của mình mà giữ dữ liệu metric còn lại; invalid sample phải còn dấu vết.

Một sample cần phân biệt bởi run, **mode**, prompt và seed. Sau đó mới tính mean, SD, sample count và missing/invalid counts. Đọc từng concept trước khi tổng hợp ba concept. SD qua prompts/seeds là mô tả tập output, không phải sai số giữa nhiều training runs độc lập.

**Trạng thái aggregate:** `aggregate_rows()` nhóm theo `(run_id, generation_mode)`, kiểm tra identity prompt×seed của protocol 8×4, và tính DINO/CLIP độc lập theo metric-specific `invalid_reason`. Vì thế batch paired 64 ảnh tạo hai hàng aggregate 32 mẫu, không trộn base với adapter. Chart cũng giữ mode riêng trong từng trace. Đây là contract đã có unit tests; chưa chứng minh metric đã chạy thành công trên batch ảnh thật. [build_report_assets.py](../scripts/build_report_assets.py) vẫn thoát với “not implemented”.

### 5.8 Human evaluation và failure analysis

[Human protocol](../configs/human_eval_v1.yaml) chọn `3 concepts × 5 configs × 4 prompt categories × 1 seed = 60 samples`, seed 11. Năm configs đại diện là n1/r16, n5/r16, n10/r16, n5/r4, n5/r32; không phải toàn bộ 18 runs nhân toàn bộ prompts.

Ít nhất 5 raters, mục tiêu 8; chấm 1–5 cho subject fidelity, prompt alignment, visual quality. Blind nghĩa là che cấu hình với người chấm, randomize thứ tự và giữ khóa ánh xạ riêng. Tool chuẩn bị packet không tự sinh ra ý kiến người thật. Phải kiểm tra selection đúng adapter mode và đủ source images trước khi phát packet.

Failure taxonomy giải thích nguyên nhân/biểu hiện mà mean score có thể che khuất: identity drift, background leakage, pose copy, prompt refusal, memorization, structure/rendering artifacts. Mỗi case cần ID mẫu và ảnh base tương ứng để phân biệt lỗi backbone với lỗi có thể do adaptation. Chi tiết: [human guide](human_evaluation.md), [rubric](evaluation_rubric.md), [taxonomy](failure_taxonomy.md).

## 6. Bản đồ thư mục: cái gì là source, cái gì là evidence?

| Vị trí | Hiểu đúng vai trò | Không được suy ra |
|---|---|---|
| `configs/` | Tham số/ma trận/version protocol | Mọi YAML đều mới, hợp lệ hoặc đã chạy |
| `data/manifests/` | Danh sách mẫu, hash, split, subset và provenance | Ảnh có trên mọi máy hoặc đã được cấp quyền public |
| `prompt_bank/` | Đề kiểm tra cố định cho các runs | Có prompt bank tức là đã sinh batch |
| `src/personalized_t2i/` | Logic reusable, có thể unit test | Tất cả module đã nối end-to-end |
| `scripts/` | CLI entry points; một số còn làm nhiều logic | Tất cả scripts đều hoạt động/đồng bộ |
| `tests/` | Kiểm tra contract, edge cases, math/control flow bằng fixtures | Test pass chứng minh LoRA đẹp hoặc GPU experiment đã chạy |
| `artifacts/` | Output và hồ sơ từng run; private/ignored | Ignored đồng nghĩa rác được xóa |
| `results/` | Bảng đã review dùng cho phân tích, kết quả dẫn xuất | CSV có header là có kết quả |
| `docs/verification/` | Nhật ký kiểm chứng theo thời điểm | Mọi note là trạng thái live hoặc không thể lỗi thời |
| `app/demo.py` | Explorer để đọc artifacts, không train | Tự tìm adapter bất kỳ ở ngoài registry/artifacts |
| `examples/`, scripts legacy/root | Upstream/vendored hoặc đường chạy lịch sử | Nên xóa vì không phải entry point chính |

Đường training chính hiện tại là `scripts/train_run.py → training/runner.py`, không nên chỉ đọc `training/train.py` rồi cho rằng đó là toàn bộ trainer. Ma trận RQ1 là `configs/ml04_sweep.yaml`; không lấy các `toy01/toy02/toy03` configs cũ làm bằng chứng rằng dataset hiện tại vẫn dùng toy IDs.

Không xóa legacy files để làm repo trông gọn hơn khi chưa có duyệt riêng. Những file này có thể giải thích nguồn gốc run cũ.

## 7. Đọc tests để biết dự án bảo vệ điều gì

| Nhóm tests | Câu hỏi mà tests giúp trả lời |
|---|---|
| [test_config](../tests/test_config.py), [test_be01](../tests/test_be01.py), [test_preflight](../tests/test_preflight.py) | Input/config sai có bị phát hiện không? |
| [test_manifest](../tests/test_manifest.py), [test_prepare](../tests/test_prepare.py) | Split/hash/nested subsets/preprocessing có tuân contract không? |
| [test_registry](../tests/test_registry.py), [test_environment](../tests/test_environment.py) | Lifecycle/provenance có đúng schema và chống ghi đè không? |
| [test_training_runner](../tests/test_training_runner.py), [test_run_sweep](../tests/test_run_sweep.py) | Chọn đúng ảnh, map CLI đúng, tạo đủ 12 ô RQ1 không? |
| [test_generation](../tests/test_generation.py) | Đủ prompt×seed×mode, metadata và RNG có đúng không? |
| [test_fidelity](../tests/test_fidelity.py), [test_alignment](../tests/test_alignment.py) | Scoring interface, normalization, invalid records, CSV upsert đúng không? |
| [test_aggregate](../tests/test_aggregate.py), [test_human_eval](../tests/test_human_eval.py), [test_failures](../tests/test_failures.py) | Bảng tổng hợp, rating protocol và case register có được kiểm tra không? |

`test_generation.py` dùng fake pipeline; metric tests có fake model/scorer. Đó là thiết kế hợp lý để test rẻ và không cần tải weights, nhưng không phải GPU evidence.

Đặc biệt [test_vertical_slice.py](../tests/test_vertical_slice.py) hiện kiểm tra config và vài fixture đơn giản; tên file không chứng minh một lượt train→generate→score thật đã chạy. Acceptance QA-01 phải dựa vào real artifacts riêng.

Tài liệu này chỉ đọc code/tests; **không báo rằng full suite vừa pass**, vì chưa chạy suite trong lượt tạo tài liệu.

## 8. Bảng DINO/CLIP, human eval, hardware, dataset link và SHA ở đâu?

### 8.1 Các bảng thực nghiệm

| Cần tìm | File | Quan sát ngày 2026-10-08 |
|---|---|---|
| DINO/CLIP từng ảnh | [metrics_per_sample.csv](../results/metrics_per_sample.csv) | Chỉ header, chưa có rows |
| Mean/SD theo run và generation mode | [metrics_aggregate.csv](../results/metrics_aggregate.csv) | Chỉ header; schema đã đồng bộ với writer, chưa có kết quả thật |
| Điểm người thật | [human_ratings.csv](../results/human_ratings.csv) | Chỉ header |
| Failure cases | [failures.csv](../results/failures.csv) | Chỉ header |
| Quy tắc chấm người | [human_eval_v1.yaml](../configs/human_eval_v1.yaml), [rubric](evaluation_rubric.md) | Đã có protocol/tooling, chưa phải ratings |

Không lấy số từ unit tests điền vào những bảng này. Per-sample schema hiện gồm `generation_mode`, và aggregate schema giữ riêng từng mode; CSV hiện chỉ là header, chưa chứa kết quả thật.

### 8.2 Hardware và thời gian

- [Local training guide](LOCAL_TRAIN_GTX1050TI.md) ghi GTX 1050 Ti 4 GB, RAM khoảng 16 GB theo quan sát ngày 2026-10-07; đây là thông tin lịch sử, không phải đo live hôm nay.
- [Colab status note](verification/COLAB_TRAINING_STATUS_2026-10-07.md) là bằng chứng Colab theo thời điểm, không chứng minh runtime đang hoạt động.
- `environment.json` của từng run là nơi cần lưu môi trường thực thi thật; [collector](../src/personalized_t2i/environment.py) triển khai việc thu thông tin.
- Runner ghi thời gian và sampled GPU memory vào `metrics.json`. `observed_peak_gpu_memory_used_mib` lấy qua nvidia-smi, có thể gồm tiến trình khác; không gọi nó là peak tensor allocation riêng của model.
- Demo local ghi metadata cạnh ảnh, trong đó có GPU/peak VRAM. Không tự chuyển số đo inference sang training wall time.

### 8.3 Các loại SHA không giống nhau

| Loại | Dùng để xác định gì? | Nơi xem |
|---|---|---|
| Git commit | Phiên bản source code | `git rev-parse HEAD`; environment của run cần giữ SHA lúc run |
| Model revision | Snapshot pretrained model | `model.revision` trong config/resolved config |
| SHA-256 từng ảnh | File ảnh có thay đổi byte không? | Cột `sha256` trong concept manifest |
| SHA-256 manifest | Danh sách/hash/split của dataset snapshot | `provenance.json` do registry tạo |
| SHA-256 dataset ZIP | Gói dataset tải về có đúng archive không? | [DATA.md](../DATA.md) |
| Trainer commit/hash | Chính xác mã trainer đã dùng | Training setup và metadata/hồ sơ của run |
| Adapter/image hash | Đúng weight/output đã dùng cho demo | Metadata JSON do [demo script](../scripts/demo_local_adapter.py) tạo |

Các revision đang có trong source, không phải xác nhận đã tải hoặc chạy tất cả:

- SD1.5: `451f4fe16113bff5a5d2269ed5ad43b0592e9a14` — [RQ1 config](../configs/ml04_sweep.yaml).
- DINOv2: `f9e44c814b77203eaa57a6bdbbd535f21ede1415` — [fidelity.py](../src/personalized_t2i/evaluation/fidelity.py).
- CLIP: `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` — [alignment.py](../src/personalized_t2i/evaluation/alignment.py).

HEAD hiện tại không nhất thiết là source commit đã train adapter 500 bước. Không điền HEAD hôm nay vào lịch sử run nếu không có bằng chứng.

## 9. Giải thích chính xác bước local pilot 500 đang làm

### 9.1 Có ba thứ dễ bị gọi chung là “pilot”

| Thiết lập | Mục đích | Có thuộc core 18 runs không? |
|---|---|---|
| `configs/pilot/dog_plush_n10_r16_ts42.yaml`: n10/r16/512, 1 update, 4 inference steps | Smoke đường trainer/reload/scorer | Không |
| Local `dog_plush_n5_r4_256_ts42`: n5/r4/256, 500 updates theo người dùng và demo docs | Khả thi trên máy local ít VRAM, qualitative pilot | Không |
| Core protocol: 3 concepts, 18 ô, 512, 500 updates, fixed evaluation bank | Trả lời RQ1/RQ2 | Có, khi artifacts và evaluation đạt contract |

Một run 256 đạt 500 bước không biến thành core 512. Sinh thêm ảnh 512 từ adapter ấy vẫn phải ghi training resolution 256, inference resolution 512; không đổi nhãn để lấp ô core.

### 9.2 Vì sao lệnh bạn vừa chạy lỗi?

Traceback dừng ở dòng import:

```text
from personalized_t2i.inference.generate import generate_evaluation_batch
ModuleNotFoundError: No module named 'personalized_t2i'
```

Package nằm trong `src/personalized_t2i/`. Interpreter đang dùng chưa tìm thấy package theo đường import hiện tại; nguyên nhân thường là chưa cài project trong environment đó hoặc chưa thêm `src` vào `PYTHONPATH`. Lỗi này xảy ra **trước khi load model/adapter**, nên không cho thấy weight 500 bước hỏng và không phải lý do train lại.

Lỗi PowerShell là vấn đề thứ hai độc lập:

- `\` cuối dòng là cách viết continuation trong shell kiểu Bash, không nối dòng như vậy trong PowerShell.
- `<run_id>` là placeholder tài liệu, không phải run ID thật; ký tự `<` gây parsing error.
- Tách `--run-id ...` thành một lệnh riêng khiến PowerShell không còn hiểu đó là argument của Python.

### 9.3 Bước kiểm tra import an toàn trước

Thực hiện ở repository root, trong environment đã dùng train thành công. Không cần cài lại cả bộ ML chỉ vì lỗi import.

```powershell
# Chỉ đổi PYTHONPATH của phiên PowerShell hiện tại, không sửa file repo.
$projectSource = (Resolve-Path -LiteralPath '.\src').Path
$env:PYTHONPATH = $projectSource
python -c "import sys, personalized_t2i; print(sys.executable); print(personalized_t2i.__file__)"
python scripts/generate_eval_set.py --help
```

Kỳ vọng package path trỏ về `src/personalized_t2i/__init__.py` của checkout này. Các lệnh trên chưa generate ảnh hay đánh giá chất lượng. Nếu đã có `PYTHONPATH` tùy chỉnh phục vụ việc khác, giữ lại giá trị cũ trước khi thay trong phiên.

### 9.4 Sửa import xong chưa chắc chạy batch ngay được

Local demo và batch evaluation có khác biệt thật trong code:

| Điều kiện | `demo_local_adapter.py` | `generate_eval_set.py` / `generate.py` |
|---|---|---|
| Mục tiêu | Một ảnh/demo pair | Fixed prompt×seed batch |
| Cache thiếu `model_index.json` | Có fallback ráp cached components | Default factory dùng `from_pretrained`, không có fallback tương đương |
| Offline | Dùng `local_files_only=True` | Default factory không đặt cờ này; có thể cần tải model |
| VRAM | CPU offload, attention slicing, VAE slicing | Default `.to(cuda)`, không tự bật offload |
| Resolution | Mặc định 256 | `inference.resolution`, mặc định 512 |
| Config | CLI/defaults của local demo | Cần resolved YAML theo contract generator/scorer |
| Metadata | Một JSON cạnh mỗi ảnh | `metadata.jsonl` của batch |

Vì vậy adapter chạy được bằng demo không tự bảo đảm batch vừa GTX 1050 Ti hoặc dùng được cùng cache. Đây là giới hạn integration cần xử lý, không phải bằng chứng weight sai.

Các kiểm tra cần hoàn tất trước chạy batch thật:

1. Adapter path đúng file đã train; không dùng literal `artifacts/<run_id>/adapter` nếu adapter thực tế nằm ngoài repo.
2. Run/config ghi đúng n5, rank4, train resolution256, tổng500, seed42 và nguồn trainer thực tế. Config smoke ban đầu 20 bước không đại diện cho run đã resume500 nếu chưa đối chiếu continuation config.
3. Evaluation settings riêng của pilot được ghi rõ, không trộn vào bảng core.
4. Cache đầy đủ hoặc loader phù hợp; đủ RAM/VRAM; không tự tải weights ngoài dự kiến.
5. Scorer tìm được `artifacts/RUN_ID/config.resolved.yaml` và `metadata.jsonl`, cùng manifest/held-out đúng phiên bản.
6. Lưu ý `--resolved-config` cho generator chỉ đọc file; nó không tự copy file ấy vào artifact directory cho scorer.

Tài liệu này không tạo config giả cho một run đã xảy ra, không gọi training hoặc batch để “thử”. Chỉ nên chuyển sang lệnh generation sau khi những input trên được xác minh.

### 9.5 Khi input đã chuẩn, command thể hiện luồng gì?

Đây là mẫu thao tác có điều kiện, không phải chứng nhận máy hiện tại đã đủ điều kiện chạy. Dùng PowerShell và mỗi lệnh Python trên một dòng; nhập giá trị thật, không nhập dấu ngoặc nhọn/góc.

```powershell
$evalRunId = Read-Host 'Run ID evaluation da duoc dang ky cho pilot'
$evalAdapter = Read-Host 'Duong dan adapter da xac minh'
$evalConfig = Read-Host 'Duong dan resolved YAML da doi chieu voi training thuc te'
if (-not (Test-Path -LiteralPath $evalAdapter -PathType Leaf)) { throw 'Khong tim thay adapter file' }
if (-not (Test-Path -LiteralPath $evalConfig -PathType Leaf)) { throw 'Khong tim thay resolved config' }

# Chay inference that; chi thuc hien sau cac preconditions o tren.
python scripts/generate_eval_set.py --run-id "$evalRunId" --adapter-path "$evalAdapter" --resolved-config "$evalConfig"
if ($LASTEXITCODE -ne 0) { throw 'Generation failed; giu output va traceback de kiem tra' }

# Chi khi batch, config trong artifact folder va held-out da hop le.
# Output pilot rieng, khong ghi vao bang core. Khong them --aggregate luc nay.
python scripts/evaluate_outputs.py --run-id "$evalRunId" --metric all --output "artifacts/$evalRunId/metrics_per_sample.csv"
```

Scoring dùng encoder weights thật, có thể cần tải DINO/CLIP nếu cache chưa có. Kiểm tra môi trường/cache trước; hướng dẫn này không cấp phép hay thực hiện tải tự động trong lượt viết tài liệu. Không chạy lại generation vào output cũ để ghi đè. Không tạo số giả khi scoring lỗi.

Sau scoring, cần kiểm tra số rows/mode, prompt×seed coverage, finite scores, invalid reasons và provenance. Chỉ có thông báo process chạy xong chưa đủ để nói evaluation hợp lệ.

## 10. Những khoảng trống hiện tại cần biết để không đọc nhầm dự án

1. **README lỗi thời:** vẫn nói mọi CLI/source/UI là placeholder, chưa có GPU experiment/model revision/manifests. Điều này không khớp code và tài liệu pilot hiện tại. Tài liệu này ghi nhận, không âm thầm sửa README.
2. **Training note có thời điểm:** `LOCAL_TRAIN_GTX1050TI.md` còn nói chưa train; `DEMO_AND_CODEX_FINAL_REVIEW.md` và người dùng đã nói hoàn tất500. Đọc theo thời điểm và loại bằng chứng, không gộp các note thành một trạng thái duy nhất.
3. **Rank sweep chưa đồng bộ:** script #16 chỉ dog_plush, model revision ngắn, manifest JSON/path `v1_n5` trong khi dữ liệu hiện tại dùng CSV v1; lookup adapter cũng khác runner. Chưa chạy script này như full core workflow. Thời gian0 cho skipped run không phải thời gian training đã đo.
4. **Aggregation chưa an toàn cho paired data:** phải tách base/adapter, kiểm tra từng identity và missing/invalid; không sửa bằng tăng count64 đơn thuần.
5. **Report CLI còn thiếu:** library chart/grid tồn tại nhưng command entry point chưa tích hợp; cần dữ liệu và schema đúng trước khi tạo figure.
6. **Bốn bảng results còn rỗng:** đủ viết methodology/tiến độ, chưa đủ kết luận tăng data/rank có lợi hay LoRA hơn base.
7. **LICENSES.md còn scaffold:** chưa có attribution hoàn chỉnh dù repo đã chứa trainer bên thứ ba. Dataset tự chụp không tự động cho phép public mọi nội dung nhìn thấy trong ảnh. Cần hoàn thiện quyền/attribution trước release.
8. **Privacy/provenance:** metadata runtime có thể chứa absolute paths. Giữ artifact riêng, scrub bản dùng công khai và giữ khả năng truy vết; không đưa raw/held-out, cache, adapter, checkpoint hoặc ảnh demo vào Git mặc định.
9. **Dry-run không nhất thiết read-only:** `run_sweep.py --dry-run` không train nhưng vẫn ghi generated configs/status report và kiểm tra dữ liệu. Không dùng nó cho một audit bị giới hạn chỉ listing/read-only.

Các vấn đề trên được đọc từ source/current files, không phải lỗi đã được sửa trong tài liệu này. Audit cũ để tham khảo: [issue progress 2026-10-07](verification/ISSUE_PROGRESS_2026-10-07.md); ưu tiên current source nếu có khác biệt.

## 11. Từ bước đang làm đến hoàn tất: tiêu chí xong cụ thể

### Chặng A — Hoàn thiện bằng chứng của local pilot

- [ ] Import đúng package trong đúng environment; chạy `--help` thành công.
- [ ] Ghi đúng adapter/model/trainer/config của run500; không đổi lịch sử run để hợp schema.
- [ ] Load adapter và sinh ảnh có metadata; nếu dùng demo pair thì giữ cùng prompt/seed/settings và nhãn qualitative pilot.
- [ ] Giải quyết loader/cache/VRAM của đường batch trước khi chạy trên máy local.
- [ ] Có batch pilot riêng đủ32 base +32 adapter, đúng8 prompts×4 seeds mỗi mode.
- [ ] Có DINO/CLIP thật và invalid/missing accounting; lưu metrics pilot riêng ngoài bảng core.
- [ ] Aggregate đã sửa/test theo mode trước khi sử dụng bảng mean/SD.

Các ô này là việc cần xác nhận tiếp, không phủ nhận những gì có thể đã tồn tại ở local nhưng chưa được kiểm tra trong lượt này.

### Chặng B — Trả lời RQ1/RQ2

- [ ] Xác nhận GPU đủ cho core512 và rank lớn; chọn budget/fallback một cách nhất quán.
- [ ] Hoàn thiện rank-sweep interface; thống nhất configs/provenance với runner.
- [ ] Chạy12 data-size cells +6 rank cells bổ sung; reuse n5/r16 đã hợp lệ, không train lặp vô ích.
- [ ] Generate/score từng run với fixed protocol; ghi cả failure và missing cells.
- [ ] Không trộn pilot256 hoặc one-step smoke vào18 core runs.

### Chặng C — Biến kết quả thành bằng chứng và báo cáo

- [ ] Thu blind ratings thật trên subset60 mẫu, đủ raters, giữ source ratings.
- [ ] Failure cases có provenance và baseline comparison, không chỉ ảnh đẹp.
- [ ] Bảng/plots RQ1 và RQ2 riêng; per-concept curves và grids cùng prompt/seed.
- [ ] Kết luận có nguồn; không claim statistical significance từ nhiều ảnh như thể là nhiều subjects độc lập.
- [ ] README/DATA/license và reproduction commands khớp current code.
- [ ] Người khác tái chạy representative run; chuẩn bị demo offline và rehearsal.

Nếu không đủ compute: báo cáo kết quả có giới hạn hoặc áp dụng contingency đã duyệt. Không dựng số để làm báo cáo trông hoàn chỉnh, không chọn các ô không đồng đều rồi gọi là controlled sweep.

## 12. Thứ tự đọc để thực sự nắm code

Đọc theo luồng, không cần đọc cả trainer nghìn dòng từ đầu:

1. [Brief](planning/01_PROJECT_BRIEF.md) và [protocol](protocol.md): mình đang trả lời câu gì?
2. [Issue backlog](planning/06_ISSUES_AND_MILESTONES.md): cần bằng chứng gì để gọi là xong?
3. [DATA](../DATA.md), [concepts](../data/manifests/concepts.csv), một manifest: model được học và được chấm bằng dữ liệu nào?
4. [RQ1 matrix](../configs/ml04_sweep.yaml), [config.py](../src/personalized_t2i/config.py): biến nào thay, biến nào giữ?
5. [train CLI](../scripts/train_run.py), [runner](../src/personalized_t2i/training/runner.py): từ config đến trainer/artifacts thế nào?
6. [registry](../src/personalized_t2i/registry.py), [environment](../src/personalized_t2i/environment.py): làm sao chứng minh đã chạy cái gì?
7. [generation](../src/personalized_t2i/inference/generate.py) và [generation test](../tests/test_generation.py): vì sao cần32 mẫu/mode và metadata?
8. [fidelity](../src/personalized_t2i/evaluation/fidelity.py), [alignment](../src/personalized_t2i/evaluation/alignment.py): điểm đo cái gì và không đo cái gì?
9. [aggregate](../src/personalized_t2i/evaluation/aggregate.py): vì sao grouping sai sẽ dẫn đến kết luận sai?
10. [human eval](human_evaluation.md), [failure taxonomy](failure_taxonomy.md), [report outline](report_outline.md): biến metric thành câu trả lời nghiên cứu thế nào?
11. [app](../app/demo.py): UI đọc lại artifact nào; phần nào chỉ là trình bày?

Khi đọc một hàm, tự trả lời: **input là gì → kiểm tra gì → xử lý gì → output ở đâu → failure được giữ thế nào → test nào kiểm tra?**

## 13. Câu trả lời ngắn khi bảo vệ đồ án

**Dự án có đề xuất thuật toán mới không?** Không. Đóng góp dự kiến là pipeline thực nghiệm có kiểm soát và phân tích ảnh hưởng data size/rank trong phạm vi đã chọn.

**Tại sao cần cả DINO và CLIP?** Vì giữ đúng subject và làm theo prompt là hai mục tiêu khác nhau; automated metrics vẫn cần human review.

**Tại sao không dùng ảnh train làm reference chính?** Để giảm đánh giá thiên về những ảnh model đã được học; held-out vẫn có hạn chế và không phải thước đo identity tuyệt đối.

**Tại sao không dùng rank càng cao càng tốt?** Rank là biến nghiên cứu; chất lượng, overfitting, adapter size và chi phí có thể đánh đổi. Phải đo chứ không chọn theo trực giác.

**Đã train500 thì hoàn thành issue nào?** Nó cung cấp bằng chứng training cho đúng run đó. Còn phải xác minh reload, metadata, settings, scoring và acceptance của issue; không tự đóng QA-01 hay core-sweep issues.

**Hiện có thể kết luận gì?** Theo thông tin người dùng và demo docs, đã có local adapter pilot500. Source có nhiều thành phần pipeline. Bốn results CSV được kiểm tra vẫn chỉ header, nên chưa có cơ sở từ những bảng này để kết luận RQ1/RQ2, chọn rank tốt nhất hoặc nói LoRA hơn base.

**Việc gần nhất cần làm là gì?** Gỡ lỗi import/cú pháp, đối chiếu hồ sơ adapter local, rồi hoàn thiện đường inference/evaluation pilot phù hợp cache và VRAM. Không cần train lại chỉ để sửa `ModuleNotFoundError`.
