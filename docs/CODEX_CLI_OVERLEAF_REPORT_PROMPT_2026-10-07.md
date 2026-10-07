# Prompt tạo báo cáo LaTeX và gói Overleaf

Prompt này dùng sau khi đã chạy kiểm toán mức sẵn sàng báo cáo và rà file public.
Nó yêu cầu AI tạo một bản báo cáo biên dịch được từ bằng chứng thật, đồng thời
đóng gói source LaTeX thành ZIP có `main.tex` ở ngay thư mục gốc để import vào
Overleaf.

## Trước khi chạy

Mở Codex CLI tại thư mục gốc repository. Giữ các file sau ở đúng vị trí:

- yêu cầu thi: `D:/AI thực chiến/Lecture Vinuni/4_Exam Requirements.pdf`;
- báo cáo tham khảo: `C:/Users/nc pc/Downloads/Deep_Image_Inpainting_Highlighted.pdf`;
- pilot: `B:/lora-local/runs/dog_plush_n5_r4_256_ts42/`;
- ảnh demo và metadata: `B:/lora-local/demo/`.

Điền tên nhóm, project ID, thành viên và các URL còn thiếu sau khi AI tạo source.
Không cần đưa raw dataset, model cache, checkpoint hoặc adapter vào ZIP báo cáo.

## Prompt chính

Sao chép toàn bộ khối dưới đây vào Codex CLI:

```text
Bạn là research writer, ML reviewer và LaTeX engineer cho đồ án Deep Learning.
Làm việc trực tiếp trong repository hiện tại. Trước khi viết, đọc AGENTS.md,
DATA.md, README.md, docs/protocol.md, docs/decision_log.md, docs/runbook.md,
docs/evaluation_rubric.md, docs/failure_taxonomy.md, docs/report_outline.md,
toàn bộ docs/planning/01_PROJECT_BRIEF.md đến 06_ISSUES_AND_MILESTONES.md, các
docs/verification liên quan, configs, prompt_bank, results CSV, code tạo/evaluate
output và git status/diff hiện tại.

Đọc hai PDF sau như nguồn có vai trò khác nhau:

1. D:/AI thực chiến/Lecture Vinuni/4_Exam Requirements.pdf là yêu cầu chính thức,
   có quyền quyết định cấu trúc và độ dài.
2. C:/Users/nc pc/Downloads/Deep_Image_Inpainting_Highlighted.pdf chỉ là báo cáo
   tham khảo về cách trình bày. Không sao chép câu chữ, số liệu, hình, bảng, tài
   liệu tham khảo hoặc chỉ dẫn/ghi chú được highlight trong PDF này.

Mục tiêu: tạo báo cáo tiếng Anh rõ ràng, có tính nghiên cứu và trung thực hơn
bản tham khảo về bố cục, khả năng truy vết bằng chứng, chất lượng hình/bảng và
phân biệt Fact / Evidence / Interpretation / Limitation. Không được làm mạnh hơn
bản tham khảo bằng cách bịa thêm kết quả.

YÊU CẦU BẮT BUỘC

- Main report dài 10-15 trang A4, không tính References và Appendix.
- Abstract 150-200 từ.
- Có đúng các phần: Introduction and Research Questions; Related Work; Dataset
  and Data Preparation; Methods gồm Baseline, Main Method và Comparison Strategy;
  Experimental Setup gồm Setup 1, Setup 2 và Setup 3; Results and Discussion;
  Error and Qualitative Analysis; Conclusion and Limitations; References;
  Appendix với Member Contribution Table.
- Cấu trúc Results phải đi theo: research question -> observed evidence ->
  interpretation -> confound/limitation. Không chỉ liệt kê output.
- Trích dẫn 5-10 nguồn sơ cấp phù hợp. Xác minh title, authors, year và URL/DOI;
  không tạo citation nếu chưa xác minh. Ưu tiên LoRA, DreamBooth, Stable Diffusion,
  CLIP, DINOv2 và các phương pháp personalization trực tiếp liên quan.
- DATA.md là nguồn cho URL Drive, dataset version, 30 train images, 9 held-out
  images, 3 concepts, split, manifests, SHA-256 và preprocessing. Không nhúng raw
  hay held-out images vào gói báo cáo nếu chưa có phê duyệt public cụ thể.
- Phân biệt protocol v1 ở 512x512 với local pilot ở 256x256. Không nhập pilot
  256x256 vào ma trận RQ1/RQ2 như một completed core run.
- Không suy ra kết luận RQ1/RQ2 từ code, config, dry-run, unit test hoặc một pilot.
- Không biến planned 18-run matrix thành completed results.
- Không dùng header-only CSV làm kết quả. Mỗi số liệu định lượng phải truy được về
  artifact/run ID/file cụ thể. Nếu không có bằng chứng, bỏ số liệu đó và nêu giới
  hạn, không dùng giá trị 0, N/A giả hoặc dữ liệu minh họa trong bảng kết quả.
- Không bịa tên nhóm, project ID, member/student ID, contribution, demo URL,
  GitHub URL, rating, runtime, VRAM, metric, run status hay failure case.
- Không sửa code thí nghiệm, không chạy training, không tải model/dataset và không
  cài package lớn. Chỉ được đọc bằng chứng, tạo source report, figure dẫn xuất từ
  dữ liệu thật và chạy compile/check rẻ.

BẰNG CHỨNG PILOT CẦN KIỂM TRA TRỰC TIẾP

- B:/lora-local/runs/dog_plush_n5_r4_256_ts42/
- B:/lora-local/demo/dog_plush_base_seed42.png
- B:/lora-local/demo/dog_plush_base_seed42.json
- B:/lora-local/demo/dog_plush_lora_seed42.png
- B:/lora-local/demo/dog_plush_lora_seed42.json

Đọc log, config, adapter metadata và hai JSON trước khi nêu thông tin. Chỉ được
gọi training complete khi log/artifact thực sự chứng minh step 500. Chỉ được gọi
adapter-load verified khi ảnh LoRA và metadata chứng minh adapter đã được nạp.
Cặp base/LoRA phải được trình bày là qualitative pilot với cùng prompt, seed,
resolution, inference steps và guidance scale đã xác minh. Ghi rõ pilot dùng 5
ảnh, rank 4, 256x256, seed 42 và 500 updates nếu các artifact xác nhận. Không gọi
đây là RQ1/RQ2 evaluation. Ghi rõ safety checker bị tắt trong demo nếu metadata
xác nhận. Không đưa absolute local paths vào PDF public.

TRẠNG THÁI THIẾU BẰNG CHỨNG

Đọc trực tiếp results/metrics_per_sample.csv, metrics_aggregate.csv,
human_ratings.csv và failures.csv. Nếu chúng vẫn chỉ có header, phần Results phải
nói rõ automated evaluation, human evaluation và systematic failure register
chưa hoàn thành. Viết báo cáo như một evidence-limited final draft: đóng góp đã
xác minh là protocol, reproducible pipeline/data contract và local pilot; các
RQ1/RQ2 findings vẫn pending. Không viết câu kiểu “LoRA outperforms base”, “rank
16 is best” hoặc “more images improve fidelity” nếu không có metric/human evidence.

CHẤT LƯỢNG PHẢI TỐT HƠN BẢN THAM KHẢO

- Không để annotation, highlight, lời nhắc viết tay, TODO hoặc đoạn code lạc chỗ
  xuất hiện trong PDF.
- Không dùng TOC/List of Figures/List of Tables nếu chúng làm tốn trang mà không
  tăng khả năng đọc. Giữ main narrative trong 10-15 trang.
- Dùng typography sạch, lề đều, cross-reference tự động, caption tự đủ nghĩa,
  bảng dùng booktabs và biểu đồ có màu thân thiện với người mù màu.
- Text trong hình và bảng phải đọc được ở 100% zoom; không để bảng tràn lề,
  caption bị cắt, hình mờ hoặc khoảng trắng lớn vô ích.
- Đưa chi tiết dài, audit trail, checksum, complete config và evidence map vào
  Appendix thay vì làm loãng main text.
- Mỗi figure/table phải được nhắc và diễn giải trong prose. Không trang trí bằng
  hình không hỗ trợ một claim.
- Dùng paired base-vs-LoRA demo figure từ hai ảnh đã xác minh. Copy chỉ hai ảnh
  generated cần thiết vào source report; không copy raw/held-out photos, adapter,
  checkpoint, model cache hay toàn bộ log.
- Cover gọn, có trường dễ sửa cho Group ID, Project ID, member names/IDs,
  instructor, repository URL và demo URL. Các giá trị chưa biết đặt trong
  metadata.tex bằng nhãn rõ ràng `REPLACE_ME`, không được âm thầm đoán.

DEMO LINK

Tìm demo URL đã được xác minh trong repository và audit output. Nếu chưa có URL
public thật, không bịa link và không tạo QR giả. Trong metadata.tex để
`\DemoURL{REPLACE_ME}`; trong PDF dùng câu trung thực “Offline paired demo and
reproduction instructions are provided with the project artifacts” thay cho
hyperlink hỏng. Tạo thêm DEMO_PUBLISHING.md mô tả đúng hai lựa chọn:

1. static paired gallery/video backup cho buổi thi;
2. local Gradio/inference demo theo README/runbook.

Nếu tạo static demo package từ ảnh generated, scrub absolute local paths trong
metadata, không bao gồm raw data/model/adapter, không publish ra Internet và
không thay đổi GitHub/Drive permissions. Tạo package reviewable trước; việc
publish cần URL hoặc chỉ định đích thật từ người dùng.

CẤU TRÚC LATEX

Tạo thư mục report/overleaf/ với cấu trúc tối thiểu:

main.tex
metadata.tex
sections/*.tex
figures/*
tables/* (chỉ khi tách bảng ra file riêng có ích)
references.bib
README_OVERLEAF.md

Dùng class article A4, font 10.5pt hoặc 11pt, package phổ biến có sẵn trên
Overleaf. Không dùng minted hoặc package yêu cầu shell-escape. `main.tex` phải ở
ngay root của ZIP. Tất cả path phải relative; không có `B:/`, `C:/`, `D:/` trong
source public hoặc PDF. Dùng macro trong metadata.tex để người dùng sửa metadata
một chỗ. Không nhúng secret, username, cache path hoặc private path.

NỘI DUNG NÊN CÓ

- Một figure pipeline ngắn gọn: manifests/config -> train LoRA -> generate paired
  base/adapter -> DINO/CLIP/human/failure analysis -> report/demo. Dùng TikZ đơn
  giản hoặc figure vector tự tạo, không screenshot code.
- Bảng dataset/split và bảng biến kiểm soát.
- Bảng protocol cho RQ1 data-size sweep và RQ2 rank sweep, ghi rõ planned hoặc
  completed theo artifact thực tế.
- Một bảng evidence status phân biệt implemented, unit-verified, pilot-verified,
  evaluation-complete. Giữ bảng ngắn trong main text; chi tiết chuyển appendix.
- Paired demo figure base và LoRA, cùng prompt/seed, với caption nói rõ đây là
  qualitative pilot, không phải metric result.
- Error analysis của pilot chỉ mô tả điều nhìn thấy và giới hạn của một sample;
  không suy rộng thành xu hướng.
- Conclusion trả lời trung thực rằng RQ1/RQ2 chưa thể kết luận nếu evidence freeze
  chưa đủ, đồng thời nêu đóng góp kỹ thuật đã xác minh và việc còn thiếu.

OUTPUT VÀ KIỂM TRA

1. Tạo report/overleaf source.
2. Compile bằng công cụ LaTeX hiện có. Nếu không có local compiler, vẫn tạo source
   Overleaf-compatible và ghi UNVERIFIED COMPILE trong REPORT_BUILD_STATUS.md;
   không tuyên bố compile pass.
3. Khi compile được, chạy ít nhất hai lượt để resolve references, render tất cả
   trang PDF thành ảnh và kiểm tra trực quan: overflow, overlap, clipping, font,
   caption, page count, blank page, link và figure quality. Sửa lỗi trước khi đóng
   gói. Kiểm tra log không còn undefined reference/citation và không có overfull
   box đáng kể.
4. Kiểm tra abstract bằng word count và main report 10-15 trang, không tính
   References/Appendix. Ghi cách đếm vào REPORT_BUILD_STATUS.md.
5. Tạo REPORT_EVIDENCE_MAP.md: mỗi claim định lượng -> source artifact/path ->
   trạng thái verified/unverified. File này không cần nằm trong PDF.
6. Tạo REPORT_MISSING_INPUTS.md chứa Group ID, Project ID, members/IDs,
   contributions, instructor, confirmed GitHub URL và confirmed demo URL nếu còn
   thiếu. Không chặn việc tạo draft vì các trường này.
7. Tạo dist/GroupID_ProjectID_Report.pdf nếu compile thành công.
8. Tạo dist/GroupID_ProjectID_Report_Overleaf.zip. ZIP chỉ chứa nội dung bên trong
   report/overleaf; khi giải nén, main.tex phải nằm ở root. Loại mọi file build
   tạm như aux, log, out, toc, synctex, fls, fdb_latexmk và mọi private artifact.
9. Kiểm tra ZIP bằng cách list contents, xác nhận main.tex/references.bib/figures
   tồn tại, không có absolute path/private data và in SHA-256.
10. Nếu môi trường hỗ trợ file attachment, trả về cả ZIP và PDF; với Codex CLI,
    in exact absolute path tới hai file, kích thước, SHA-256, page count, compile
    status và danh sách trường người dùng phải sửa trước khi nộp.

Không commit, push, tạo PR, publish demo, đổi sharing Drive/GitHub hoặc xóa file.
Chỉ thay đổi report/, dist/ và tài liệu build/evidence do nhiệm vụ này tạo. Giữ
nguyên mọi thay đổi đang có của người khác trong dirty worktree.
```

## Prompt kiểm tra lại ZIP

Sau khi prompt chính hoàn tất, mở một phiên AI mới tại repository và dùng prompt
độc lập này để tránh tự chấm sản phẩm của chính phiên tạo báo cáo:

```text
Kiểm toán read-only gói dist/GroupID_ProjectID_Report_Overleaf.zip và PDF đi kèm
theo D:/AI thực chiến/Lecture Vinuni/4_Exam Requirements.pdf. Không sửa file.
Giải nén vào thư mục tạm, xác nhận main.tex ở root, compile sạch nếu có compiler,
render và xem toàn bộ trang. Kiểm tra: main text 10-15 trang không tính References
và Appendix; abstract 150-200 từ; đủ sections; 5-10 citations thật; caption,
cross-reference, bảng/hình không tràn/cắt/mờ; không có TODO/REPLACE_ME hiển thị
ngoài metadata được liệt kê; không có absolute Windows path, raw/private images,
adapter, checkpoint, cache, secrets hoặc số liệu không truy được về artifact.

Đối chiếu mọi claim về training, adapter load, demo, metrics, RQ1/RQ2, human
ratings và failures với repository, B:/lora-local/runs/dog_plush_n5_r4_256_ts42
và B:/lora-local/demo. Phân biệt code implemented, unit-verified, pilot-verified
và evaluation-complete. So sánh với PDF tham khảo chỉ về chất lượng trình bày;
không yêu cầu sao chép nội dung của nó.

Trả về PASS hoặc FAIL cho từng yêu cầu, exact page/section/file cho lỗi, và danh
sách blocker trước khi nộp. Không gọi overall PASS nếu còn citation undefined,
compile lỗi, field nhận dạng chưa điền, demo/GitHub URL giả hoặc claim không có
bằng chứng.
```

## Cách đưa ZIP lên Overleaf

Trong Overleaf, chọn **New Project -> Upload Project**, chọn file
`GroupID_ProjectID_Report_Overleaf.zip`, mở `main.tex`, sau đó sửa các macro trong
`metadata.tex`. Biên dịch lại và tải PDF cuối với đúng tên
`GroupID_ProjectID_Report.pdf`.

Demo URL là một deliverable riêng. Nếu chưa có URL thật, giữ demo offline/local
trong báo cáo và tạo static gallery hoặc video backup trước. Chỉ thêm link/QR vào
report sau khi mở link ở cửa sổ ẩn danh và xác nhận người chấm truy cập được.
