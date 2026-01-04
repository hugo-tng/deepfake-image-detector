import os
from PIL import Image
import torch

from utils.config import GlobalConfig, LabelConfig
from data.datasets import get_transforms
from data.facecrop import FaceCropper
import matplotlib.pyplot as plt


def preprocess_image(image_path, img_size: 224, device, cropper: FaceCropper = None):
    """
    Đọc và tiền xử lý ảnh giống hệt như lúc training (Resize -> Normalize).
    """
    if cropper is not None:
        # Cropper tự handle việc đọc file và trả về PIL Image
        pil_image = cropper(image_path)
        status_msg = "Processed Input (Face Crop)"
    else:
        # Fallback nếu không dùng crop
        pil_image = Image.open(image_path).convert('RGB')
        status_msg = "Original Input (No Crop)"

    plt.figure(figsize=(4, 4))
    plt.imshow(pil_image)
    plt.axis('off')
    plt.title(status_msg, fontsize=10)
    plt.show()

    if not os.path.exists(image_path):
        raise FileNotFoundError(f" Không tìm thấy ảnh tại: {image_path}")

    # 2. Lấy bộ transform - test
    transforms_dict = get_transforms(img_size)
    transform = transforms_dict['test']

    # 3. Biến đổi ảnh thành tensor
    img_tensor = transform(pil_image)

    # 4. Thêm chiều batch (C, H, W) -> (1, C, H, W)
    img_tensor = img_tensor.unsqueeze(0)

    return img_tensor.to(device)

def predict_image(models_list, image_path):
    """
    Hàm suy luận (Inference) cho ảnh đầu vào.
    
    Args:
        models: Danh sách mô hình 
        image_path: Đường dẫn ảnh
    """  
    face_cropper = FaceCropper(
        out_size=GlobalConfig.CROP_SIZE,         
        target_face_ratio=1.2 
    )

    # 1. Tiền xử lý
    print(f"\n--- Processing: {os.path.basename(image_path)} ---")
    img_tensor = preprocess_image(image_path, GlobalConfig.IMG_SIZE, GlobalConfig.DEVICE, cropper=face_cropper)

    results_data = []

    for entry in models_list:
        model = entry['model']
        model_name = entry['name']
        result = model_predict(model, img_tensor, model_name)
        results_data.append(result)

    # 3. In kết quả
    print(f"{'MODEL NAME':<25} | {'PREDICTION':<12} | {'CONFIDENCE':<12} | {'FAKE PROB':<12}" + 
          f" | {'SPATIAL W':<12} | {'FREQ W':<12}")
    print("-" * 100)

    for row in results_data:
        spatial_w = row.get('Spatial Weight', None)
        freq_w = row.get('Frequency Weight', None)

        spatial_w = f"{spatial_w:.4f}" if isinstance(spatial_w, (float, int)) else "_"
        freq_w = f"{freq_w:.4f}" if isinstance(freq_w, (float, int)) else "_"

        print(
            f"{row.get('Model Name', '_'):<25} | "
            f"{row.get('Prediction', '_'):<12} | "
            f"{row.get('Confidence', '_'):<12} | "
            f"{row.get('Fake Prob', '_'):<12} | "
            f"{spatial_w:<12} | "
            f"{freq_w:<12}"
        )
    
    print("-" * 100)

    return results_data


def model_predict(model, image_tensor, model_name):
    model.eval()
    
    with torch.no_grad():
        # Forward pass
        outputs = model(image_tensor)
        
        # Tính xác suất (Softmax)
        probs = torch.softmax(outputs, dim=1)

        # Lấy class có xác suất cao nhất
        confidence, pred_class_idx = torch.max(probs, 1)

        conf_score = confidence.item()
        label = LabelConfig.ID2LABELS.get(pred_class_idx.item(), "Unknown")
    
        # Lấy xác suất cụ thể của lớp AI-Generated (Fake)
        fake_prob = probs[0][LabelConfig.FAKE_IDX].item()

        # Lấy trọng số kết quả
        spatial_w, freq_w = None, None

        if hasattr(model, "get_feature_importance"):
            try:
                spatial_w, freq_w = model.get_feature_importance(image_tensor)

                spatial_w = spatial_w.mean().item()
                freq_w = freq_w.mean().item()

            except Exception:
                spatial_w, freq_w = None, None

        result = {
            "Model Name": model_name,
            "Prediction": label.upper(),
            "Confidence": f"{conf_score*100:.2f}%",
            "Fake Prob": f"{fake_prob*100:.2f}%",
            "Spatial Weight": spatial_w,
            "Frequency Weight": freq_w
        }

    return result