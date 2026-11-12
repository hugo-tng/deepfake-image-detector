"""Typed, immutable runtime configuration loaded from project.yaml."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping
import copy

import torch
import yaml

SOURCE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = SOURCE_ROOT.parent
PROJECT_CONFIG_PATH = SOURCE_ROOT / "configs" / "project.yaml"


@dataclass(frozen=True)
class LabelsConfig:
    real_id: int = 0
    fake_id: int = 1
    real_name: str = "Real"
    fake_name: str = "Fake"

    @property
    def id_to_label(self) -> dict[int, str]:
        return {self.real_id: self.real_name, self.fake_id: self.fake_name}

    @property
    def label_to_id(self) -> dict[str, int]:
        return {self.real_name: self.real_id, self.fake_name: self.fake_id}


@dataclass(frozen=True)
class PathsConfig:
    workspace_root: Path
    source_root: Path
    datasets_root: Path
    outputs_root: Path
    raw_real_dir: Path
    raw_fake_dir: Path
    processed_real_dir: Path
    processed_fake_dir: Path
    splits_dir: Path

    @property
    def train_csv(self) -> Path: return self.splits_dir / "train.csv"
    @property
    def val_csv(self) -> Path: return self.splits_dir / "val.csv"
    @property
    def test_csv(self) -> Path: return self.splits_dir / "test.csv"


@dataclass(frozen=True)
class RuntimeConfig:
    device: torch.device
    seed: int
    deterministic: bool = False


@dataclass(frozen=True)
class DatasetConfig:
    image_size: int
    crop_size: int
    csv_columns: tuple[str, ...] = ("path", "label")
    valid_labels: tuple[int, ...] = (0, 1)
    class_directories: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({"real": "Real", "fake": "Fake"}))


@dataclass(frozen=True)
class ColorJitterConfig:
    brightness: float = 0.4
    contrast: float = 0.4
    saturation: float = 0.2
    hue: float = 0.05


@dataclass(frozen=True)
class GaussianBlurConfig:
    probability: float = 0.25
    kernel_size: int = 3
    sigma: tuple[float, float] = (0.1, 1.0)


@dataclass(frozen=True)
class JPEGConfig:
    probability: float = 0.3
    quality_range: tuple[int, int] = (70, 95)


@dataclass(frozen=True)
class GaussianNoiseConfig:
    probability: float = 0.1
    std: float = 0.01


@dataclass(frozen=True)
class TransformConfig:
    resize_multiplier: float = 1.15
    horizontal_flip_probability: float = 0.5
    color_jitter: ColorJitterConfig = field(default_factory=ColorJitterConfig)
    gaussian_blur: GaussianBlurConfig = field(default_factory=GaussianBlurConfig)
    jpeg_compression: JPEGConfig = field(default_factory=JPEGConfig)
    gaussian_noise: GaussianNoiseConfig = field(default_factory=GaussianNoiseConfig)
    imagenet_normalization: bool = True


@dataclass(frozen=True)
class LoaderConfig:
    batch_size: int = 16
    num_workers: int = 2
    pin_memory_when_cuda: bool = True
    drop_last_train: bool = True
    persistent_workers: bool = True


@dataclass(frozen=True)
class TrainingDefaults:
    num_epochs: int = 10
    learning_rate: float = 1e-4
    backbone_lr_multiplier: float = 0.1
    weight_decay: float = 3e-4
    freeze_epochs: int = 3
    label_smoothing: float = 0.05
    use_amp: bool = True
    scheduler_type: str = "cosine"
    scheduler_t_max: int = 10
    scheduler_eta_min: float = 1e-6
    early_stopping_monitor: str = "validation_f1"
    early_stopping_patience: int = 7
    log_interval: int = 10
    save_best_only: bool = True


@dataclass(frozen=True)
class ModelConfig:
    name: str
    mode: str
    architecture: str
    parameters: Mapping[str, Any]


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig
    training: TrainingDefaults
    output_dir: Path

    @property
    def checkpoint_dir(self) -> Path: return self.output_dir / "checkpoints"
    @property
    def log_dir(self) -> Path: return self.output_dir / "logs"
    @property
    def OUTPUT_DIR(self) -> str: return str(self.output_dir)
    @property
    def CHECKPOINT_DIR(self) -> str: return str(self.checkpoint_dir)
    @property
    def LOG_DIR(self) -> str: return str(self.log_dir)
    # Compatibility properties used by the current Trainer/supporter code.
    @property
    def MODE(self) -> str: return self.model.mode
    @property
    def MODEL_NAME(self) -> str: return self.model.name
    @property
    def MODEL_CONFIG(self) -> dict[str, Any]: return dict(self.model.parameters)
    @property
    def FREEZE_EPOCHS(self) -> int: return self.training.freeze_epochs
    @property
    def NUM_EPOCHS(self) -> int: return self.training.num_epochs
    @property
    def LABEL_SMOOTHING(self) -> float: return self.training.label_smoothing
    @property
    def USE_AMP(self) -> bool: return self.training.use_amp
    @property
    def LEARNING_RATE(self) -> float: return self.training.learning_rate
    @property
    def WEIGHT_DECAY(self) -> float: return self.training.weight_decay
    @property
    def SCHEDULER_TYPE(self) -> str: return self.training.scheduler_type
    @property
    def T_MAX(self) -> int: return self.training.scheduler_t_max
    @property
    def EARLY_STOPPING_PATIENCE(self) -> int: return self.training.early_stopping_patience
    @property
    def LOG_INTERVAL(self) -> int: return self.training.log_interval
    @property
    def SAVE_BEST_ONLY(self) -> bool: return self.training.save_best_only
    def create_directories(self):
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ProjectConfig:
    labels: LabelsConfig
    paths: PathsConfig
    runtime: RuntimeConfig
    dataset: DatasetConfig
    transforms: TransformConfig
    loader: LoaderConfig
    training: TrainingConfig
    models: Mapping[str, ModelConfig]

    def for_model(self, name: str) -> ExperimentConfig:
        if name not in self.models:
            raise ValueError(f"Unknown model '{name}'. Available models: {sorted(self.models)}")
        return ExperimentConfig(self.models[name], self.training, self.paths.outputs_root / name)


def _pair(value: Any, cast): return tuple(cast(item) for item in value)


def load_config(path: Path | str = PROJECT_CONFIG_PATH) -> ProjectConfig:
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    labels = raw.get("labels", {})
    reference = raw.get("reference_parameters", {})
    paths_raw = raw.get("paths", {})
    datasets = WORKSPACE_ROOT / paths_raw.get("datasets", "Datasets")
    outputs = WORKSPACE_ROOT / paths_raw.get("outputs", "Outputs")
    processed = datasets / "Face_Cropped"
    paths = PathsConfig(SOURCE_ROOT.parent, SOURCE_ROOT, datasets, outputs,
                        datasets / "Real", datasets / "AI_Generated",
                        processed / "Real", processed / "Fake", datasets / "Split_Data")
    image = reference.get("image", {})
    loader_raw = reference.get("loader", {})
    repro = reference.get("reproducibility", {})
    aug = reference.get("augmentation", {})
    cj, blur, jpeg, noise = aug.get("color_jitter", {}), aug.get("gaussian_blur", {}), aug.get("jpeg_compression", {}), aug.get("gaussian_noise", {})
    training_raw, optimizer, scheduler, early = reference.get("training", {}), reference.get("optimizer", {}), reference.get("scheduler", {}), reference.get("early_stopping", {})
    gpu_runtime = raw.get("profiles", {}).get("gpu", {}).get("reference_runtime", {})
    training = TrainingDefaults(int(training_raw.get("num_epochs", 10)), float(optimizer.get("learning_rate", 1e-4)), float(optimizer.get("backbone_lr_multiplier", 0.1)), float(optimizer.get("weight_decay", 3e-4)), int(training_raw.get("freeze_epochs", 3)), float(training_raw.get("label_smoothing", 0.05)), bool(training_raw.get("use_amp", gpu_runtime.get("use_amp", True))), str(scheduler.get("type", "cosine")), int(scheduler.get("t_max", 10)), float(scheduler.get("eta_min", 1e-6)), str(early.get("monitor", "validation_f1")), int(early.get("patience", 7)))
    models = {}
    base = raw.get("models", {}).get("efficientnet", {}).get("model_config", {})
    for name, item in raw.get("models", {}).items():
        params = copy.deepcopy(base); params.update(item.get("model_config", {})); params.update(item.get("overrides", {}))
        models[name] = ModelConfig(name, item.get("mode", "hybrid"), item.get("architecture", ""), MappingProxyType(params))
    return ProjectConfig(LabelsConfig(int(labels.get("real", {}).get("id", 0)), int(labels.get("fake", {}).get("id", 1)), str(labels.get("real", {}).get("display_name", "Real")), str(labels.get("fake", {}).get("display_name", "Fake"))), paths, RuntimeConfig(torch.device("cuda" if torch.cuda.is_available() else "cpu"), int(repro.get("random_seed", 42)), bool(repro.get("deterministic_notebook_setting", False))), DatasetConfig(int(image.get("image_size", 240)), int(image.get("crop_size", 256))), TransformConfig(float(aug.get("resize_multiplier", 1.15)), float(aug.get("horizontal_flip_probability", 0.5)), ColorJitterConfig(**{k: float(cj.get(k, v)) for k, v in {"brightness": .4, "contrast": .4, "saturation": .2, "hue": .05}.items()}), GaussianBlurConfig(float(blur.get("probability", .25)), int(blur.get("kernel_size", 3)), _pair(blur.get("sigma", [.1, 1.]), float)), JPEGConfig(float(jpeg.get("probability", .3)), _pair(jpeg.get("quality_range", [70, 95]), int)), GaussianNoiseConfig(float(noise.get("probability", .1)), float(noise.get("std", .01)))), LoaderConfig(int(loader_raw.get("batch_size", 16)), int(loader_raw.get("num_workers", 2)), bool(loader_raw.get("pin_memory_when_cuda", True)), bool(loader_raw.get("drop_last_train", True)), int(loader_raw.get("num_workers", 2)) > 0), training, MappingProxyType(models))


_PROJECT = load_config()

# Compatibility adapters for existing notebooks/modules.
class LabelConfig:
    REAL_IDX, FAKE_IDX = _PROJECT.labels.real_id, _PROJECT.labels.fake_id
    REAL_NAME, FAKE_NAME = _PROJECT.labels.real_name, _PROJECT.labels.fake_name
    ID2LABELS, LABELS2ID = _PROJECT.labels.id_to_label, _PROJECT.labels.label_to_id

class GlobalConfig:
    DEVICE, IMG_SIZE, CROP_SIZE = _PROJECT.runtime.device, _PROJECT.dataset.image_size, _PROJECT.dataset.crop_size
    RANDOM_SEED, NUM_WORKERS, BATCH_SIZE = _PROJECT.runtime.seed, _PROJECT.loader.num_workers, _PROJECT.loader.batch_size

class PathConfig:
    BASE_DIR, DATASETS, OUTPUTS = str(_PROJECT.paths.workspace_root), str(_PROJECT.paths.datasets_root), str(_PROJECT.paths.outputs_root)
    RAW_AI_GEN_IMGS, RAW_REAL_IMGS = str(_PROJECT.paths.raw_fake_dir), str(_PROJECT.paths.raw_real_dir)
    PROCESSED_DATA, REAL_IMGS, AI_GEN_IMGS, SPLITTED_DATASETS = str(_PROJECT.paths.processed_real_dir.parent), str(_PROJECT.paths.processed_real_dir), str(_PROJECT.paths.processed_fake_dir), str(_PROJECT.paths.splits_dir)
    train_csv, val_csv, test_csv = str(_PROJECT.paths.train_csv), str(_PROJECT.paths.val_csv), str(_PROJECT.paths.test_csv)

class LegacyTrainingConfig:
    def __init__(self, mode: str, model_name: str):
        experiment = _PROJECT.for_model(model_name); t = experiment.training
        self.MODE, self.MODEL_NAME = mode, model_name
        self.OUTPUT_DIR, self.CHECKPOINT_DIR, self.LOG_DIR = str(experiment.output_dir), str(experiment.checkpoint_dir), str(experiment.log_dir)
        self.NUM_EPOCHS, self.LEARNING_RATE, self.WEIGHT_DECAY = t.num_epochs, t.learning_rate, t.weight_decay
        self.FREEZE_EPOCHS, self.LABEL_SMOOTHING, self.EARLY_STOPPING_PATIENCE = t.freeze_epochs, t.label_smoothing, t.early_stopping_patience
        self.USE_AMP, self.SCHEDULER_TYPE, self.T_MAX = t.use_amp, t.scheduler_type, t.scheduler_t_max
        self.LOG_INTERVAL, self.SAVE_BEST_ONLY, self.MODEL_CONFIG = t.log_interval, t.save_best_only, dict(experiment.model.parameters)
    def create_directories(self):
        Path(self.OUTPUT_DIR).mkdir(parents=True, exist_ok=True); Path(self.CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True); Path(self.LOG_DIR).mkdir(parents=True, exist_ok=True)

TrainingConfigLegacy = LegacyTrainingConfig
TrainingConfig = LegacyTrainingConfig

