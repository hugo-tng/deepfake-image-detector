import torch
import os
import random
import numpy as np
from models.detector import DeepfakeDetector
from utils.config import TrainingConfig, GlobalConfig

def set_seed(seed: int = 42, deterministic: bool = True):
    """
    Set random seed for full reproducibility.
    
    Args:
        seed (int): Random seed
        deterministic (bool): If True, enforce deterministic behavior (slower)
    """
    # Python
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # CuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Torch >= 1.8
        torch.use_deterministic_algorithms(True)
    else:
        # Faster but non-deterministic
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    print(f"[INFO] Random seed set to {seed} | deterministic={deterministic}")


def build_model(config: TrainingConfig):
    """
    Factory function để khởi tạo model.
    
    Args:
        config: Class TrainingConfig chứa cấu hình
        mode (str, optional): Nếu muốn override mode trong config (vd: test nhánh lẻ)
    
    Returns:
        model: Mô hình đã được đẩy lên Device (GPU/CPU)
    """
    # Sử dụng mode từ tham số hoặc từ config
    selected_mode = config.MODE

    print(f"🛠️ Building Model | Mode: {selected_mode} | Device: {GlobalConfig.DEVICE}")

    model = DeepfakeDetector(
        mode=selected_mode,
        **config.MODEL_CONFIG
    )

    model.to(GlobalConfig.DEVICE)
    return model

def load_weights(model, config: TrainingConfig):
    """
    Load trọng số từ file .pth vào model một cách an toàn.
    Tự động xử lý trường hợp key chứa 'module.' (do train DataParallel).
    
    Args:
        model: Kiến trúc model đã khởi tạo
        checkpoint_path: Đường dẫn đến file .pth
        device: Torch device
    
    Returns:
        model: Model đã load trọng số và chuyển sang eval mode
        info: Dict chứa thông tin thêm (epoch, metrics...)
    """
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(config.CHECKPOINT_DIR):
        raise FileNotFoundError(f"❌ Không tìm thấy file trọng số tại: {checkpoint_path}")
        
    print(f"🔄 Loading weights from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=GlobalConfig.DEVICE)
    
    # Lấy state_dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict'] # Checkpoint đầy đủ
    else:
        state_dict = checkpoint # Chỉ lưu state_dict
        
    # Xử lý key 'module.' (nếu train nhiều GPU)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k 
        new_state_dict[name] = v
        
    # Load vào model
    try:
        model.load_state_dict(new_state_dict, strict=True)
    except RuntimeError as e:
        print(f"⚠️ Warning: Key mismatch (Strict loading failed). Retrying with strict=False.")
        print(f"Error detail: {e}")
        model.load_state_dict(new_state_dict, strict=False)
        
    model.to(GlobalConfig.DEVICE)
    model.eval() # chuyển sang eval mode khi load
    
    print("✅ Weights loaded successfully!")
    
    # Trả về thông tin epoch/metrics nếu có
    info = {
        'epoch': checkpoint.get('epoch', -1),
        'metrics': checkpoint.get('metrics', {})
    }
    return model, info


def count_parameters(model):
    """Hàm phụ trợ: Đếm số lượng tham số trainable"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📊 Model Summary:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    return total_params, trainable_params