# Codex CLI prompts for issues #19–#24

These prompts are intentionally split at issue #21. The first makes the existing
evaluation/explorer/reporting code verifiable. The second consumes that contract
to prepare the report, reproduction, and presentation deliverables. Run them in
order or in separate worktrees after integrating the first result.

The task framing follows the official Codex prompting guidance: give the agent a
concrete objective, repository constraints, acceptance criteria, and verification
requirements while allowing it to inspect the implementation before editing:
https://developers.openai.com/cookbook/examples/gpt-5/codex_prompting_guide

## Prompt A — issues #19, #20, and #21

```text
Bạn đang làm việc trong repository
sharkdownwindows/Personalized-Text-to-Image-Generation-with-LoRA.
Mục tiêu: hoàn thiện phần code và verification có thể làm cục bộ cho issues
#19 EVAL-05, #20 FE-01 và #21 REP-01. Hãy sửa code thật, chạy tests nhẹ, và
báo cáo bằng chứng; không dừng ở kế hoạch.

Trước khi sửa:
1. Đọc AGENTS.md và docs/planning/01_PROJECT_BRIEF.md đến
   docs/planning/06_ISSUES_AND_MILESTONES.md.
2. Đọc live issues #19, #20, #21 và kiểm tra main hiện tại; snapshot audit dùng
   commit 5da945cd04 nhưng phải kiểm tra drift.
3. Đọc docs/verification/ISSUE_PROGRESS_2026-10-07.md, implementation, tests,
   git status và diff. Không ghi đè, stash, reset, restore hoặc trộn thay đổi
   chưa commit của người khác. Nếu checkout đang bẩn, dùng worktree riêng.
4. Tái hiện gap trước khi sửa. Giữ configs/ là source of truth, logic reusable
   trong src/, scripts/ chỉ là entry point mỏng.

Hiện trạng cần kiểm chứng, không được tin mù quáng:
- #19 đã merge taxonomy/validator, nhưng results/failures.csv chỉ có header;
  chưa có representative case thật.
- #20 đã merge app/demo.py, nhưng chưa có reusable tests chứng minh browse,
  same-prompt/same-seed comparison và provenance trên artifact contract thật.
- #21 đã merge reporting/charts.py và reporting/grids.py, nhưng
  scripts/build_report_assets.py vẫn thoát với “not implemented yet”; chưa có
  figures thật trong repo và aggregate schema có thể không khớp reader.
- #17 aggregation có bug đã biết với base + adapter rows. Không sửa #17 ngoài
  thay đổi compatibility tối thiểu bắt buộc cho #21; nếu cần redesign schema,
  ghi blocker rõ thay vì lén mở rộng scope.

#19 — EVAL-05:
- Giữ taxonomy hiện có; không thay tên tag nếu không có migration rõ ràng.
- Bổ sung tests/validation tối thiểu còn thiếu để mỗi case truy được tới
  run/concept/prompt/seed và LoRA image; attribution base_model/lora/both phải
  có paired base output cùng prompt và seed, full model revision; memorization
  phải có nearest training image.
- Phát hiện duplicate case/tag và provenance mismatch thay vì chỉ kiểm tra field
  không rỗng. Giữ needs_review/uncertain khi evidence chưa đủ.
- CLI validation phải read-only, fail rõ với header-only CSV, missing file và
  inconsistent pair. Không sinh case giả hoặc tự điền metric.
- Nếu không có outputs thật, chỉ hoàn thiện workflow/tests và ghi EVAL-05 là
  evidence pending; không đánh dấu acceptance criterion đã đạt.

#20 — FE-01:
- Giữ Gradio mỏng, không thêm training qua UI, API server, database hoặc queue.
- Tách helper thuần nếu cần để test bằng fixture artifact nhỏ mà không launch
  server hoặc tải model.
- Test: discover concept/run/config; chỉ hiển thị run hợp lệ; load precomputed
  image; compare đúng cùng prompt+seed giữa hai runs; hiện source path/run/config/
  prompt/seed/model revision; lỗi rõ khi metadata/image/config thiếu hoặc hỏng.
- Bảo đảm UI không import/call training runner trong luồng browse. Nếu live
  generation chưa có artifact/model thì precomputed explorer là đủ cho P0.

#21 — REP-01:
- Nối scripts/build_report_assets.py vào reusable charts.py và grids.py; không
  viết lại logic đã merge.
- CLI nhận explicit input/output paths, đọc source CSV/JSONL mà không sửa nguồn,
  và trả exit code khác 0 khi input trống, schema sai, run thiếu hoặc không đủ
  comparable conditions. Không tạo figure rỗng rồi coi là thành công.
- RQ1 chỉ so n={1,3,5,10}, rank=16. RQ2 chỉ so rank={4,16,32}, n=5. Hiện từng
  concept; không gộp base và adapter thành một series nếu schema có mode.
- Grid phải chọn deterministic từ frozen prompt bank/seeds và dùng cùng
  prompt+seed qua các conditions; không chọn theo metric/ảnh đẹp.
- Mỗi figure/grid phải lưu sidecar manifest hoặc metadata chứa source path/hash,
  run IDs, concept, prompt/seed khi applicable, và thời điểm tạo.
- Thêm tests bằng synthetic fixture cho success và incomplete/malformed inputs.
  Kiểm tra downstream Gradio vẫn đọc được contract.

Definition of done:
- Thay đổi nhỏ, đúng scope; không refactor unrelated code.
- Chạy targeted tests cho failures, explorer helpers, charts/grids/CLI và
  compatibility; sau đó chạy full cheap suite nếu environment cho phép.
- Chạy git diff --check. Báo chính xác command, pass/fail và limitation.
- Thêm verification notes riêng cho #19/#20/#21, phân biệt rõ
  implemented/tested với real experiment evidence pending.
- Không download model, cài ML package lớn, chạy GPU/training, sửa Colab đang
  chạy, commit raw/private data, adapters/checkpoints/generated batches.
- Không fabricate figures, metrics, ratings, failure cases hoặc research claims.
- Không commit, push, tạo PR, comment/close issue nếu người dùng chưa yêu cầu
  trong phiên Codex CLI đó.
```

## Prompt B — issues #21, #22, #23, and #24

```text
Bạn đang làm việc trong repository
sharkdownwindows/Personalized-Text-to-Image-Generation-with-LoRA.
Mục tiêu: dùng contract/report assets của #21 để hoàn thiện phần code, tài liệu
và verification có thể làm cục bộ cho #21 REP-01, #22 DOC-01, #23 QA-02 và
#24 DOC-02. Hãy triển khai tối thiểu nhưng end-to-end; không bịa kết quả để
đánh dấu issue hoàn tất.

Trước khi sửa:
1. Đọc AGENTS.md, docs/planning/01–06, live issues #21–#24, git status/diff,
   docs/verification/ISSUE_PROGRESS_2026-10-07.md và output của Prompt A nếu có.
2. Kiểm tra branch/main drift. Không ghi đè thay đổi chưa commit của người khác;
   dùng worktree riêng nếu cần.
3. Xác nhận scripts/build_report_assets.py, reporting schema, Gradio artifact
   contract, results/*.csv, prompt bank và run metadata trước khi thiết kế.
4. Lập bảng “acceptance criterion → artifact/evidence path → verified/pending”.
   Không dùng code presence thay cho experiment evidence.

#21 — REP-01 foundation:
- Nếu Prompt A đã hoàn tất và tests pass, reuse contract; không viết lại.
- Nếu chưa, chỉ bổ sung phần tối thiểu còn thiếu: CLI tạo RQ1/RQ2 charts và
  same-prompt/same-seed grids từ explicit real inputs, validation fail-loud,
  provenance sidecar có source hash/run IDs/prompt/seed.
- Không tạo dummy figures trong results/ và không coi synthetic fixtures là
  evidence nghiên cứu.

#22 — DOC-01:
- Nâng docs/report_outline.md thành report draft có cấu trúc rõ cho RQ1 và RQ2,
  methods, experiment coverage, quantitative results, qualitative evidence,
  failure analysis, human evaluation, limitations, confounds và claim boundary.
- Tạo evidence map machine-readable hoặc Markdown liên kết mỗi main claim tới
  table/figure/run IDs/source data. Với data chưa có, ghi PENDING + exact input
  cần thiết; không viết kết luận xu hướng giả.
- Report phải nêu fixed-compute confound, chỉ một training seed nếu đúng, ba
  concepts, metric limitations, missing/failed runs và disagreement giữa metric/
  human review. Null result phải được giữ.
- Nếu hợp lý, thêm script/check nhỏ kiểm tra broken/missing evidence links;
  không xây document framework lớn.

#23 — QA-02:
- Viết independent reproduction runbook/checklist dùng một representative config
  từ clean environment, ghi operator khác ML-02 author, source commit, config
  hash, dataset hash, environment, commands, timestamps, output paths,
  nondeterminism/differences và verdict.
- Tạo preflight/dry-run validation nhẹ nếu giúp phát hiện missing dependency/data/
  model access trước GPU; reuse code hiện có, không tạo trainer thứ hai.
- Không chạy/download model hoặc GPU trong task này. Không ghi “reproduced” khi
  chưa có người độc lập chạy train → load → generate → score thành công.
- Verification note phải có trạng thái ready-for-execution hoặc blocked, cùng
  exact command người dùng cần chạy và evidence cần trả về.

#24 — DOC-02:
- Tạo presentation outline/source nhỏ gọn: problem, protocol, coverage, RQ1,
  RQ2, failures, limitations, reproducibility, demo flow, Q&A.
- Tạo offline-demo manifest/checker hoặc static-gallery builder chỉ từ artifacts
  có thật; fail rõ nếu thiếu. Không commit private/large images hoặc video.
- Chuẩn bị live-demo checklist và fallback checklist; training không chạy trong
  demo. Ghi expected technical questions và evidence-backed answers; để PENDING
  cho số liệu chưa có.
- Không tuyên bố rehearsal/video/gallery đã hoàn tất khi chưa có artifact và
  reviewer xác nhận.

Definition of done:
- #21 có executable report-assets path hoặc blocker cụ thể.
- #22 có report draft + evidence map không chứa invented findings.
- #23 có executable reproduction protocol nhưng trạng thái chỉ complete sau
  real independent run.
- #24 có slide/demo sources và validators; completion cần real offline backup và
  rehearsal evidence.
- Thêm verification notes riêng cho #21–#24 với implemented/tested/pending.
- Chạy targeted tests/checkers và git diff --check; báo đúng kết quả, kể cả fail.
- Không download model, cài ML package lớn, chạy GPU/training, sửa Colab, tạo
  metrics/ratings giả, commit private data/weights, hoặc xóa legacy files.
- Không commit, push, tạo PR, comment/close issue nếu người dùng chưa yêu cầu
  trong phiên Codex CLI đó.
```
