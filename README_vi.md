# Phát hiện ảnh chân dung do AI bằng học sâu

> English version: [README.md](README.md)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2B-red)
![Trạng thái](https://img.shields.io/badge/trang_thai-nghiên_cứu-green)
![Giấy phép](https://img.shields.io/badge/license-giáo_dục-orange)

Dự án nghiên cứu PyTorch phân loại ảnh chân dung thành **người thật** hoặc **do AI tạo**. Dự án so sánh các mô hình một nhánh với mô hình lai kết hợp đặc trưng không gian và tần số.

## Mục lục

- [Giới thiệu](#giới-thiệu)
- [Tính năng chính](#tính-năng-chính)
- [Dữ liệu](#dữ-liệu)
- [Kiến trúc mô hình](#kiến-trúc-mô-hình)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Cài đặt](#cài-đặt)
- [Cách sử dụng](#cách-sử-dụng)
- [Tóm tắt kết quả](#tóm-tắt-kết-quả)
- [Artifact và cấu hình](#artifact-và-cấu-hình)

## Giới thiệu

Ảnh chân dung do AI tạo có thể rất giống ảnh chụp. Dự án khảo sát việc kết hợp cấu trúc hình ảnh với các mẫu trong miền tần số có giúp phân biệt ảnh thật và ảnh tạo hay không. Mã nguồn bao gồm chuẩn bị dữ liệu, huấn luyện, đánh giá, suy luận mẫu và trực quan hóa bằng notebook Jupyter cùng các module Python.

## Tính năng chính

- **Hai nhánh bổ trợ:** EfficientNet-B1 trích xuất đặc trưng không gian; CNN dựa trên FFT phân tích đặc trưng tần số.
- **Attention fusion bất đối xứng:** hợp nhất 512 đặc trưng không gian và 256 đặc trưng tần số bằng cổng học theo từng chiều.
- **So sánh baseline:** hỗ trợ mô hình chỉ dùng không gian, chỉ dùng tần số, hybrid bất đối xứng và hybrid đối xứng (512 + 512).
- **Quy trình huấn luyện:** chuẩn hóa ImageNet, crop/lật và biến đổi màu, nén JPEG, nhiễu Gaussian; đóng băng backbone không gian lúc khởi động rồi fine-tune, early stopping theo F1 validation.
- **Artifact đánh giá:** lưu metrics, confusion matrix, ROC curve, lịch sử huấn luyện và checkpoint.

## Dữ liệu

Dự án dùng ảnh chân dung thật và ảnh do AI tạo. Notebook chuẩn bị dữ liệu crop/resize ảnh rồi tạo các CSV train, validation và test theo phân tầng. Nhãn là `0 = Real`, `1 = Fake`.

Notebook tham chiếu [AI-image-detection-FAKE](https://huggingface.co/datasets/fcsn37/AI-image-detection-FAKE) và [AI-image-detection-REAL](https://huggingface.co/datasets/fcsn37/AI-image-detection-REAL). Nguồn gốc và quyền sử dụng phụ thuộc bộ dữ liệu gốc; hãy kiểm tra điều khoản trước khi dùng lại hoặc phân phối.

## Kiến trúc mô hình

Mô hình lai kết hợp hai góc nhìn của ảnh RGB 240 × 240: EfficientNet-B1 trích xuất đặc trưng không gian, còn FFT kết hợp CNN trích xuất đặc trưng tần số. Attention gate hợp nhất hai nhánh trước khi bộ phân loại dự đoán Real hoặc Fake.

![Sơ đồ kiến trúc mô hình](assets/model_architecture.svg)

Xem [`diagram_code.md`](diagram_code.md) để biết sơ đồ Mermaid chi tiết.

## Cấu trúc dự án

```text
Source/
├── assets/                 # Sơ đồ kiến trúc
├── data/                   # Dataset, biến đổi ảnh, loader, crop mặt
├── engine/                 # Huấn luyện, đánh giá, suy luận
├── models/                 # Module không gian, tần số, fusion, detector
├── notebooks/              # Chuẩn bị dữ liệu, huấn luyện, kiểm tra, trực quan
├── configs/                # Cấu hình tham chiếu và manifest dữ liệu
├── tools/                  # Bộ kiểm tra cấu hình dự án chỉ-đọc
├── utils/                  # Cấu hình, metrics, checkpoint, biểu đồ
├── diagram_code.md         # Sơ đồ Mermaid chi tiết
├── requirements.txt
├── README.md
└── README_vi.md
```

## Cài đặt

Dùng Python 3.10 trở lên trong virtual environment. Chạy lệnh từ `Source/`:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install jupyter
jupyter lab
```

Mở Jupyter tại `Source/` để `system_path.py` tìm thấy module. Torchvision có thể tải trọng số EfficientNet ImageNet ở lần chạy đầu. Mã tự chọn CUDA nếu có, nếu không chạy bằng CPU. Nếu bản PyTorch được ghim không phù hợp phần cứng, hãy cài bản tương thích.

### Clone dự án và kiểm tra cấu hình

Sau khi clone, giữ `Source/`, `Datasets/` và `Outputs/` (nếu có) là các thư mục cùng cấp. Tải thủ công dataset, kiểm tra giấy phép, rồi đặt các CSV split và ảnh crop `Real/`, `Fake/` vào `Datasets/`.

Từ `Source/`, kiểm tra cấu trúc trước khi mở notebook:

```bash
python tools/validate_config.py --profile experiment
```

Dùng `cpu` hoặc `gpu` thay cho `experiment` để kiểm tra profile máy. Lệnh chỉ-đọc; checkpoint lịch sử thiếu chỉ là cảnh báo. Thêm `--strict-hardware` nếu muốn yêu cầu CUDA cho profile GPU.

## Cách sử dụng

Chạy notebook theo thứ tự khi cần:

1. [`01_DataPrepare.ipynb`](notebooks/01_DataPrepare.ipynb) — kiểm tra, tiền xử lý và chia ảnh; chỉnh đường dẫn nguồn nếu cần.
2. [`02_TrainingModel.ipynb`](notebooks/02_TrainingModel.ipynb) — huấn luyện bốn biến thể. Checkpoint và log được ghi vào `Outputs/<tên-mô-hình>/` ở thư mục gốc.
3. [`03_ModelTesting.ipynb`](notebooks/03_ModelTesting.ipynb) — tải checkpoint tốt nhất và dự đoán ảnh mẫu. Thay các đường dẫn mẫu viết sẵn cho máy của bạn.
4. [`04_VisualizeExample.ipynb`](notebooks/04_VisualizeExample.ipynb) — trực quan FFT và phổ tần số sau lọc; cập nhật đường dẫn ảnh mẫu.

`FaceCrop.ipynb` là notebook thử nghiệm riêng dùng Hugging Face/Google Colab.

## Tóm tắt kết quả

Bảng được chép từ `Outputs/*/logs/evaluation_results/test_metrics.json`. Đây là kết quả đã lưu trên cùng tập test 4.000 ảnh; Fake là lớp dương khi tính precision, recall và F1. Kết quả không đảm bảo hiệu suất trên bộ dữ liệu khác.

| Mô hình                         |   Accuracy |  Precision |     Recall |         F1 |    ROC AUC |
| ------------------------------- | ---------: | ---------: | ---------: | ---------: | ---------: |
| Không gian (EfficientNet-B1)    |     0.9793 |     0.9660 |     0.9935 |     0.9795 |     0.9975 |
| Tần số (FFT-CNN)                |     0.8800 |     0.8172 |     0.9790 |     0.8908 |     0.9562 |
| Hybrid bất đối xứng (512 + 256) | **0.9815** | **0.9684** | **0.9955** | **0.9818** | **0.9980** |
| Hybrid đối xứng (512 + 512)     |     0.9768 |     0.9600 |     0.9950 |     0.9772 |     0.9973 |

Trong các lần chạy đã lưu, hybrid bất đối xứng có accuracy, F1 và ROC AUC cao nhất. Mức cải thiện so với baseline không gian nhỏ và chỉ áp dụng cho tập chia này.

## Artifact và cấu hình

[`configs/project.yaml`](configs/project.yaml) ghi đường dẫn di động, profile, biến thể mô hình và các tham số tham chiếu. [`configs/dataset_manifest.yaml`](configs/dataset_manifest.yaml) nêu nguồn dữ liệu, yêu cầu CSV và artifact lịch sử mong đợi. Chúng chỉ phục vụ tài liệu và kiểm tra; code train và notebook hiện vẫn dùng `utils/config.py` như cũ.

`Outputs/` chứa checkpoint và kết quả lịch sử. Không commit dataset, weights, output mới sinh, local override hoặc validation report.

## Giấy phép

Dự án được phát hành cho mục đích giáo dục và nghiên cứu theo các điều khoản trong [`LICENSE`](LICENSE).
