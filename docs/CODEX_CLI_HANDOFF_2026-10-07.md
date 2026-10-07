# Prompt for Codex CLI

Copy the following prompt into Codex CLI in a separate checkout. This prompt
does not authorize starting training, downloading models, or publishing changes.
Task context and explicit acceptance criteria follow the official guidance:
https://learn.chatgpt.com/guides/best-practices

```text
Bạn là engineering partner cho university LoRA project này. Hãy sửa code thật,
kiểm tra bằng tests nhẹ và báo cáo diff; không dừng ở đề xuất.

Đọc AGENTS.md, docs/planning/01–06 và
docs/verification/ISSUE_PROGRESS_2026-10-07.md. Đọc lại issues live #16, #17,
#21 của sharkdownwindows/Personalized-Text-to-Image-Generation-with-LoRA và
implementation hiện tại. Snapshot audit dùng main 5da945cd04; kiểm tra drift.

Codex Desktop đang sửa #13–#15 và các file runner.py, training/sweep.py,
scripts/run_sweep.py, scripts/run_vertical_slice.py, configs/ml04_sweep.yaml,
alignment.py cùng tests liên quan. Không sửa cùng checkout hoặc ghi đè diff
của Desktop. Dùng checkout/worktree riêng theo conventions. Nếu checkout này
đang được Desktop sử dụng, dừng mutation tại đây và chuẩn bị checkout riêng.
Không tự stash/reset/restore/cherry-pick các thay đổi chưa commit của người khác.

Ưu tiên 1 — #17: sửa aggregation và completeness.
- Tái hiện lỗi: 32 base + 32 adapter cùng run bị báo 64 > 32.
- Tách thống kê theo run/concept/generation_mode và checkpoint khi cần;
  không trộn base và adapter, không chỉ tăng expected_sample_count thành 64.
- 8 prompts × 4 seeds = 32 mẫu/mode; dùng prompt/seed versioned để kiểm tra
  missing/duplicate/unexpected identities, không chỉ so số lượng hàng.
- Đọc contract per-sample writer hiện có. Định nghĩa metric-specific validity
  để không tính invalid metric, vẫn giữ metric hợp lệ khi metric khác thất bại.
  Reproduce row valid=False với finite score đang bị tính mean. Không đổi
  policy tùy ý làm mất partial scores; thêm field/policy tối thiểu nếu cần.
- Không biến missing/NaN thành 0. Giữ mean, descriptive SD, valid/invalid/missing
  count rõ nghĩa; giữ rows theo concept và provenance. Báo missing run toàn bộ
  theo expected run matrix, không âm thầm bỏ run thiếu.
- Tương thích charts/Gradio và CSV hiện có: kiểm tra reader thực tế trước khi
  đổi schema; legacy rows thiếu mode không được silently gán sai baseline/LoRA.
- Không sửa CLIP/DINO inference implementations hoặc CLI scoring đang thuộc
  Desktop. Có thể thêm aggregate-only entry point mỏng nếu CLI hiện tại buộc
  load model trước khi aggregate.

Ưu tiên 2 — #21: nối CLI tạo charts/grids hiện có.
- scripts/build_report_assets.py hiện chỉ báo chưa implemented: nối reusable
  reporting/charts.py và reporting/grids.py, tránh viết lại logic đã có.
- Đồng bộ schema aggregate; chọn mode minh bạch, không trộn baseline/adapter.
- RQ1: n=1/3/5/10 rank16. RQ2: rank4/16/32 n5. Hiện từng concept.
- Grid same-prompt/same-seed, lựa chọn deterministic từ prompt bank;
  không chọn ảnh đẹp theo metric. Ghi source table/hash, run IDs, prompt/seed.
- Input trống/thiếu/invalid phải báo rõ; không tạo fake result figures.
- Kiểm tra cả downstream Gradio với fixture, giữ training ngoài UI.

Ưu tiên 3 — #16: chỉ triển khai sau khi interface #15 ổn định.
- scripts/run_ml05_sweep.py hiện hardcode dog_plush, short revision 451f4fe,
  v1_n5 folders/JSON manifest và checkpoint path cũ. Sửa theo CSV manifests
  và config-driven runner đã ổn định từ Desktop; không tạo second trainer.
- Ba concepts: cat_mug, dog_plush, blue_white_vase; n5; ranks4/16/32;
  alpha=rank; ts42; full model SHA; giữ các controlled variables của v1.
- Reuse n5/r16 từ data sweep nếu provenance/config/artifact tương thích.
  Chỉ 6 additional trainings khi 3 anchors đã hoàn tất: tổng core 18 unique.
- Completed run không overwrite/rerun; running không khởi động lại;
  failed/corrupt/incomplete phải lưu lỗi và quyết định retry rõ ràng.
- Adapter size/time lấy từ artifact contract/metrics thật. Missing để null
  hoặc explicit unknown, không ghi 0 giây/0MB rồi coi thành công.
- Có dry-run/plan kiểm tra matrix mà không launch GPU. Logic trong src,
  scripts entrypoint mỏng; không xóa/move legacy files chưa được cho phép.
- Nếu #15 chưa ổn định, hoàn tất #17/#21 và báo dependency #16; đừng tự
  refactor shared sweep/runner đang được Desktop sửa.

Verification/definition of done:
- Tests synthetic có ý nghĩa cho two modes 32+32, partial/invalid scores,
  duplicate/missing prompt-seed và whole-run missing; source CSV không đổi.
- Fixture charts/grids truy đúng mode/run/prompt/seed, CLI không load models.
- Rank plan ba concepts, anchor reuse/mismatch, running/failed/malformed state
  và measurement missing. Không train để chạy unit tests.
- Chạy tests cần thiết + git diff --check. Báo chính xác command/result.
- Thêm verification notes riêng cho #16/#17/#21, phân biệt implemented/tested
  với real experiment evidence pending. Giữ failed/unfavorable evidence.
- Không download models, cài ML packages lớn, khởi chạy GPU/training, sửa
  notebook/runtime đang train, commit/push/PR hay đóng issues. Không bịa
  metrics, human ratings hoặc kết luận nghiên cứu. Chỉ source/config/tests/docs.
- Cuối cùng báo thay đổi, tests, tương thích downstream, phạm vi chưa làm,
  và lệnh Colab để người dùng chạy sau khi Desktop tích hợp xong.
```
