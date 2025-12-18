import os
import torch

class PathConfig:
    BASE_DIR = os.path.join(os.pardir, os.pardir)
    DATASETS = os.path.join(BASE_DIR, 'Datasets')
    RAW_AI_GEN_IMGS = os.path.join(DATASETS, 'AI_Generated')
    RAW_REAL_IMGS = os.path.join(DATASETS, 'Real')

    PROCESSED_DATA = os.path.join(DATASETS, "processed")
    REAL_IMGS = os.path.join(PROCESSED_DATA, 'Real')
    AI_GEN_IMGS = os.path.join(PROCESSED_DATA, "Fake")
    
    SPILTTED_DATASETS = os.path.join(DATASETS, 'Split_Data')
    OUTPUTS = os.path.join(BASE_DIR, 'Outputs')

    # file paths
    train_csv = os.path.join(SPILTTED_DATASETS, 'train.csv')
    val_csv = os.path.join(SPILTTED_DATASETS, 'val.csv')
    test_csv = os.path.join(SPILTTED_DATASETS, 'test.csv')


class TrainingConfig:
    def __init__(self, mode: str, model_name: str):
        # --- General settings ---
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.RANDOM_SEED = 42

        # --- class parameters ---
        self.MODE = mode
        self.MODEL_NAME = model_name
        
        # --- Output paths ---
        self.OUTPUT_DIR = os.path.join(PathConfig.OUTPUTS, self.MODEL_NAME)
        self.CHECKPOINT_DIR = os.path.join(self.OUTPUT_DIR, "checkpoints")
        self.LOG_DIR = os.path.join(self.OUTPUT_DIR, "logs")

        # --- Hyperparameters ---
        self.NUM_EPOCHS = 10
        self.BATCH_SIZE = 16
        self.LEARNING_RATE = 1e-4
        self.WEIGHT_DECAY = 1e-4
        self.IMG_SIZE = 240
        self.NUM_WORKERS = 0
        self.FREEZE_EPOCHS = 3 

        # Early stopping
        self.EARLY_STOPPING_PATIENCE = 10

        # Mixed precision training
        self.USE_AMP = True  

        # Scheduler
        self.SCHEDULER_TYPE = 'cosine'
        self.T_MAX = 50

        # Logging
        self.LOG_INTERVAL = 50
        self.SAVE_BEST_ONLY = True

        # --- Model parameters ---
        self.MODEL_CONFIG = {
            'num_classes': 2,
            'dropout_rate': 0.4,
            'efficientnet_model': 'efficientnet_b1',
            'spatial_dim': 512,
            'freq_dim': 512,
            'use_attention_fusion': True,
            'attention_hidden_dim': 256
        }

    def create_directories(self):
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(self.LOG_DIR, exist_ok=True)

