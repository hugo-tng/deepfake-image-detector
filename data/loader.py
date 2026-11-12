from torch.utils.data import DataLoader
from .datasets import DeepFakeDataset
from utils.config import GlobalConfig, LoaderConfig, _PROJECT

def get_data_loader(
        ds: DeepFakeDataset, 
        shuffle: bool, is_train: bool=False,
        loader_config: LoaderConfig | None = None,
    ) -> DataLoader:
    config = loader_config or _PROJECT.loader
    return DataLoader(
        ds, batch_size=config.batch_size, shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=(GlobalConfig.DEVICE.type == 'cuda' and config.pin_memory_when_cuda),
        persistent_workers=(config.persistent_workers and config.num_workers > 0 and is_train),
        drop_last=(config.drop_last_train and is_train)
    )
