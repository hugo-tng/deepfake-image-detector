import pandas as pd
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class AddGaussianNoise:
    def __init__(self, mean=0., std=0.01):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std + self.mean
        tensor = tensor + noise
        return torch.clamp(tensor, 0.0, 1.0)

class JPEGCompression:
    def __init__(self, quality_range=(70, 95)):
        self.quality_range = quality_range

    def __call__(self, img):
        # Chọn ngẫu nhiên quality trong range
        quality = random.randint(*self.quality_range)

        # Nếu input là PIL Image -> Convert sang Numpy (OpenCV)
        if isinstance(img, Image.Image):
            img_np = np.asarray(img)
            
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif isinstance(img, np.ndarray):
            img_np = img
        else:
            return img # Fallback nếu format lạ
        
        # Áp dụng nén JPEG với OpenCV
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encimg = cv2.imencode('.jpg', img_np, encode_param)
        decimg = cv2.imdecode(encimg, cv2.IMREAD_COLOR)

        # Convert ngược lại PIL để tương thích torchvision
        decimg = cv2.cvtColor(decimg, cv2.COLOR_BGR2RGB)
        return Image.fromarray(decimg)
    

def get_transforms(img_size=240):
    """
    Trả về dictionary chứa transform cho train và val/test
    """
    # Chuẩn hóa theo ImageNet (Standard practice)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(int(img_size * 1.15)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),

            # Color augmentation (spatial robustness)
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.05
            ),

            # Blur (camera / codec artifact)
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
            ], p=0.25),

            # JPEG compression (deepfake-realistic)
            transforms.RandomApply([
                JPEGCompression(quality_range=(70, 95))
            ], p=0.3),

            transforms.ToTensor(),

            # Sensor noise (rất nhẹ)
            transforms.RandomApply([
                AddGaussianNoise(std=0.01)
            ], p=0.1),
            
            normalize
        ]),
        'val': transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize
        ]),
        'test': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize
        ]),
    }
    return data_transforms

# Dataset cho DeepFake
class DeepFakeDataset(Dataset):
    def __init__(self, data: pd.DataFrame, transform=None):
        """
        data: pandas DataFrame chứa đường dẫn ảnh và nhãn
        transform: torchvision transforms
        """
        self.paths = data["path"].tolist()
        self.labels = data["label"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # Lấy đường dẫn ảnh và nhãn
        img_path = self.paths[idx]
        label = self.labels[idx]

        try:
            # Load ảnh
            img = cv2.imread(img_path)
            if img is None:
                raise RuntimeError(f"Không thể đọc ảnh: {img_path}")
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(img)

        except Exception as e:
           raise RuntimeError(f"Lỗi load ảnh: {img_path}") from e
        
        # Áp dụng transform
        if self.transform:
            image = self.transform(image)

        return image, label