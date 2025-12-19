# 🔬 Cell Segmentation Project: U-Net vs. StarDist

Dự án này tập trung giải quyết bài toán phân đoạn tế bào (Cell Segmentation) trong ảnh y sinh, đặc biệt xử lý các trường hợp tế bào dính nhau và dữ liệu thưa (sparse data). Dự án so sánh và triển khai hai phương pháp tiếp cận: **Semantic Segmentation (U-Net)** và **Instance Segmentation (StarDist)**.

## 🚀 Tính năng nổi bật

1.  **Chiến lược Tiền xử lý thông minh (Smart Preprocessing):**
    * **Adaptive Normalization:** Tự động cân bằng sáng dựa trên phân vị (percentile), giúp làm rõ tế bào trong ảnh tối/mờ.
    * **Multiscale Smart ROI:** Kỹ thuật cắt ảnh (crop) tập trung vào vùng có tế bào thay vì cắt ngẫu nhiên vào vùng nền đen, kết hợp với zoom đa tỉ lệ để tăng cường dữ liệu.

2.  **Đa dạng Mô hình:**
    * **U-Net (Baseline):** Phân đoạn nhị phân (Nền vs. Tế bào).
    * **StarDist (Advanced):** Mô hình định hướng đối tượng, sử dụng các tia hình sao (star-convex polygons) để tách rời các tế bào dính nhau.

3.  **Hệ thống Đánh giá chuẩn xác:**
    * Tích hợp bộ đánh giá **mAP (Mean Average Precision)** theo chuẩn cuộc thi Data Science Bowl 2018.
    * Hỗ trợ tính toán tại nhiều ngưỡng IoU (0.5 - 0.95).
    * Tự xây dựng thuật toán hậu xử lý (NMS, Polygon reconstruction) không phụ thuộc thư viện ngoài.

## 📂 Cấu trúc thư mục

```text
project_root/
├── data/                       # Chứa dữ liệu (stage1_train, stage1_test)
├── src/                        # Mã nguồn chính
│   ├── config.py               # Cấu hình toàn bộ (Hyperparameters, Paths)
│   ├── dataset.py              # Dataset cho U-Net (Binary Mask)
│   ├── dataset_stardist.py     # Dataset cho StarDist (Distance Map & Prob Map)
│   ├── modelUnet.py            # Kiến trúc U-Net
│   ├── model_stardist.py       # Kiến trúc StarDist (với Softplus activation)
│   ├── train.py                # Script huấn luyện U-Net
│   ├── train_stardist.py       # Script huấn luyện StarDist (Mixed Precision)
│   ├── evaluation.py           # Đánh giá mAP cho U-Net
│   ├── evaluation_stardist.py  # Đánh giá mAP cho StarDist (Custom NMS)
│   ├── transforms.py           # Augmentation (Albumentations)
│   └── utils.py                # Hàm phụ trợ (Seed, Visualize...)
├── notebooks/                  # Các file Jupyter Notebook chạy thử
├── best_unet_checkpoint.pth    # Weight tốt nhất của U-Net
├── best_stardist_checkpoint.pth # Weight tốt nhất của StarDist
└── README.md
```
## Để triển khai mô hình vui lòng cài đặt các thư viện cần thiết trong file requirements.txx
Sau đó tiến hành các bước cấu hình và huấn luyện
1. Cấu hình

Mở file src/config.py để chỉnh sửa đường dẫn dữ liệu và tham số:
Python

class Config:
    TRAIN_DIR = './data/train'
    BATCH_SIZE = 8  # Giảm xuống 4 nếu VRAM < 4GB
    IMG_SIZE = 256
    CROP_STRATEGY = 'multiscale_smart' # Chiến lược crop ảnh thông minh

2. Huấn luyện (Training)

Train U-Net (Cơ bản):
Bash

python src/train.py

Train StarDist (Nâng cao): Lưu ý: StarDist sử dụng Mixed Precision (AMP) để tiết kiệm bộ nhớ GPU.
Bash

python src/train_stardist.py

3. Đánh giá (Evaluation)

Tính toán chỉ số mAP trên tập Validation và vẽ biểu đồ Precision-IoU.

Đánh giá U-Net:
Bash

python src/evaluation.py

Đánh giá StarDist:
Bash

python src/evaluation_stardist.py
