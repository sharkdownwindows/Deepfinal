import os
import shutil
from pathlib import Path

def main():
    source_pool = Path("data/raw/dog_plush/v1/train_pool")
    if not source_pool.exists():
        print(f"Không tìm thấy thư mục nguồn: {source_pool}")
        return

    # Lấy danh sách tất cả các file ảnh trong train_pool gốc
    all_images = list(source_pool.glob("*.jpeg")) + list(source_pool.glob("*.jpg")) + list(source_pool.glob("*.png"))
    print(f"Tìm thấy tổng số {len(all_images)} ảnh trong thư mục gốc.")

    # Các mức data size cần tạo
    data_sizes = [1, 3, 5, 10]

    for n in data_sizes:
        target_dir = Path(f"data/raw/dog_plush/v1_n{n}/train_pool")
        target_dir.mkdir(parents=True, exist_ok=True)

        # Lấy đúng n ảnh đầu tiên (hoặc ngẫu nhiên)
        selected_images = all_images[:n]
        
        # Copy ảnh sang thư mục tương ứng
        for img_path in selected_images:
            shutil.copy(img_path, target_dir / img_path.name)
            
        print(f"Đã tạo và copy {len(selected_images)} ảnh vào: {target_dir}")

if __name__ == "__main__":
    main()