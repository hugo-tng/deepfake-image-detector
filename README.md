# Deepfake Detection using Dual-Branch Hybrid Architecture

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.13%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

## Giới thiệu (Introduction)

Dự án này đề xuất một kiến trúc học sâu mới (**Hybrid Asymmetric Architecture**) nhằm giải quyết thách thức trong việc phát hiện ảnh Deepfake chất lượng cao.

Thay vì chỉ dựa vào thông tin hình ảnh (RGB), mô hình kết hợp song song hai luồng xử lý:

1.  **Spatial Branch (Miền không gian):** Sử dụng EfficientNet-B1 để nắm bắt ngữ nghĩa và cấu trúc khuôn mặt.
2.  **Frequency Branch (Miền tần số):** Sử dụng biến đổi Fourier (FFT) và các bộ lọc thông cao để phát hiện các dấu vết nhân tạo (artifacts) bất thường mà mắt thường không thấy được.

Hai luồng thông tin được hợp nhất thông qua cơ chế **Residual Attention Fusion**, cho phép mô hình tự động học trọng số tối ưu cho từng nhánh.

## Tính năng nổi bật (Key Features)

- **Multi-modal Analysis:** Kết hợp phân tích đa miền (Spatial + Frequency).
- **Asymmetric Design:** Thiết kế bất đối xứng (Spatial 512-dim, Frequency 256-dim) giúp tối ưu hóa tài nguyên tính toán và giảm nhiễu.
- **Robust Preprocessing:** Tích hợp mô phỏng nén ảnh (JPEG Compression) và nhiễu (Gaussian Noise) để tăng độ bền vững.
- **Two-stage Training:** Chiến lược huấn luyện 2 giai đoạn (Frozen & Fine-tuning) giúp hội tụ ổn định.
- **High Performance:** Đạt độ chính xác >99% trên tập dữ liệu kiểm thử hỗn hợp.
