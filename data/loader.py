from torch.utils.data import DataLoader
from .datasets import DeepFakeDataset
from utils.config import TrainingConfig

def get_data_loader(ds: DeepFakeDataset, cfg: TrainingConfig, shuffle: bool) :
    return DataLoader(
        ds, batch_size=cfg.BATCH_SIZE, shuffle=shuffle, 
        num_workers=cfg.NUM_WORKERS, pin_memory=False
    )