import os
import torch

class LabelConfig:
    # Class labels indices
    REAL_IDX = 0
    FAKE_IDX = 1

    # Class labels names
    REAL_NAME = 'Real'
    FAKE_NAME = 'Fake'

    # Label mappings
    ID2LABELS = {
        REAL_IDX: REAL_NAME,
        FAKE_IDX: FAKE_NAME
    }

    LABELS2ID = {
        REAL_NAME: REAL_IDX,
        FAKE_NAME: FAKE_IDX
    }


class GlobalConfig:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    IMG_SIZE = 240
    CROP_SIZE = 256
    RANDOM_SEED = 42
    NUM_WORKERS = 2
    BATCH_SIZE = 16



class PathConfig:
    BASE_DIR = os.path.join(os.pardir, os.pardir)
    DATASETS = os.path.join(BASE_DIR, 'Datasets')
    RAW_AI_GEN_IMGS = os.path.join(DATASETS, 'AI_Generated')
    RAW_REAL_IMGS = os.path.join(DATASETS, 'Real')

    PROCESSED_DATA = os.path.join(DATASETS, 'Face_Cropped')
    REAL_IMGS = os.path.join(PROCESSED_DATA, 'Real')
    AI_GEN_IMGS = os.path.join(PROCESSED_DATA, "Fake")
    
    SPLITTED_DATASETS = os.path.join(DATASETS, 'Split_Data')
    OUTPUTS = os.path.join(BASE_DIR, 'Outputs')

    # file paths
    train_csv = os.path.join(SPLITTED_DATASETS, 'train.csv')
    val_csv = os.path.join(SPLITTED_DATASETS, 'val.csv')
    test_csv = os.path.join(SPLITTED_DATASETS, 'test.csv')


class TrainingConfig:
    def __init__(self, mode: str, model_name: str):
        # --- class parameters ---
        self.MODE = mode
        self.MODEL_NAME = model_name
        
        # --- Output paths ---
        self.OUTPUT_DIR = os.path.join(PathConfig.OUTPUTS, self.MODEL_NAME)
        self.CHECKPOINT_DIR = os.path.join(self.OUTPUT_DIR, "checkpoints")
        self.LOG_DIR = os.path.join(self.OUTPUT_DIR, "logs")

        # --- Hyperparameters ---
        self.NUM_EPOCHS = 10
        self.LEARNING_RATE = 1e-4
        self.WEIGHT_DECAY = 3e-4
        self.FREEZE_EPOCHS = 3
        self.LABEL_SMOOTHING = 0.05

        # Early stopping
        self.EARLY_STOPPING_PATIENCE = 7

        # Mixed precision training
        self.USE_AMP = True  

        # Scheduler
        self.SCHEDULER_TYPE = 'cosine'
        self.T_MAX = 10

        # Logging
        self.LOG_INTERVAL = 10
        self.SAVE_BEST_ONLY = True

        # --- Model parameters ---
        self.MODEL_CONFIG = {
            'num_classes': 2,
            'dropout_rate': 0.4,
            'efficientnet_model': 'efficientnet_b1',
            'spatial_dim': 512,
            'freq_dim': 256,
            'use_attention_fusion': True,
            'attention_hidden_dim': 256
        }

    def create_directories(self):
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(self.LOG_DIR, exist_ok=True)

