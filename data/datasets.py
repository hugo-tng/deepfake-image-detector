import pandas as pd
import random
import io
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
        
        if isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)
        
        quality = random.randint(*self.quality_range)
        with io.BytesIO() as buffer:
            img.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            return Image.open(buffer).convert("RGB")
    

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

            # JPEG compression (deepfake-realistic)
            transforms.RandomApply([
                JPEGCompression(quality_range=(70, 95))
            ], p=0.3),

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

class DeepFakeDataset(Dataset):
    def __init__(self, data: pd.DataFrame, transform=None):
        """
        data: pandas DataFrame chứa đường dẫn ảnh và nhãn
        transform: torchvision transforms
        """
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Lấy row
        row = self.data.iloc[idx]
        img_path = row["path"]
        label = row["label"]

        try:
            # Load ảnh
            image = Image.open(img_path).convert("RGB")

        except Exception as e:
           raise RuntimeError(f"Lỗi load ảnh: {img_path}") from e
        
        # Áp dụng transform
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)