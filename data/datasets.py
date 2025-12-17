import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

def get_transforms(img_size=224):
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
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            normalize
        ]),
        'val': transforms.Compose([
            transforms.Resize((img_size, img_size)),
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

            if image is None:
                raise ValueError(f"Không thể tải ảnh tại đường dẫn: {img_path}")


            # Áp dụng transform
            if self.transform:
                image = self.transform(image)

            return image, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            print(f"Lỗi khi xử lý ảnh {img_path}: {e}")
            
            # Trả về một ảnh rỗng và nhãn -1 để đánh dấu lỗi
            dummy_image = Image.new("RGB", (224, 224))  # Kích thước tùy ý
            if self.transform:
                dummy_image = self.transform(dummy_image)
            return dummy_image, torch.tensor(-1, dtype=torch.long)