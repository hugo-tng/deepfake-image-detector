from torch.utils.data import DataLoader
from .datasets import DeepFakeDataset
from utils.config import TrainingConfig

def get_data_loader(
        ds: DeepFakeDataset, cfg: TrainingConfig, 
        shuffle: bool, is_train: bool=False
    ) -> DataLoader:
    
    return DataLoader(
        ds, batch_size=cfg.BATCH_SIZE, shuffle=shuffle, 
        num_workers=cfg.NUM_WORKERS, 
        pin_memory=(cfg.DEVICE.type == 'cuda'), 
        persistent_workers=(cfg.NUM_WORKERS > 0 and is_train)
    )