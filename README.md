# AI-Generated Portrait Detection with Deep Learning

> Vietnamese version: [README_vi.md](README_vi.md)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2B-red)
![Status](https://img.shields.io/badge/status-research-green)
![License](https://img.shields.io/badge/license-educational-orange)

A PyTorch research project that classifies portrait images as **real** or **AI-generated**. It compares single-branch baselines with hybrid spatial-frequency models.

## Table of Contents

- [Introduction](#introduction)
- [Key Features](#key-features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results Summary](#results-summary)
- [Artifacts and Configuration](#artifacts-and-configuration)

## Introduction

AI-generated portraits can resemble photographs closely. This project studies whether combining visual structure with frequency-domain patterns helps distinguish real and generated portraits. The repository includes data preparation, training, evaluation, inference examples, and visualization as Jupyter notebooks and reusable Python modules.

## Key Features

- **Two complementary branches:** EfficientNet-B1 extracts spatial features; an FFT-based CNN analyzes frequency features.
- **Asymmetric attention fusion:** combines 512 spatial features and 256 frequency features with learned element-wise gates.
- **Baseline comparison:** supports spatial-only, frequency-only, asymmetric hybrid, and symmetric hybrid (512 + 512) experiments.
- **Robust training pipeline:** ImageNet normalization, crop/flip and color augmentation, JPEG compression and Gaussian noise; spatial backbone warm-up freezing, fine-tuning, and early stopping on validation F1.
- **Evaluation artifacts:** saves metrics, confusion matrices, ROC curves, training history, and checkpoints.

## Dataset

The project uses real and AI-generated portrait images. The preparation notebook crops/resizes them and creates stratified train, validation, and test CSV files. Labels are `0 = Real` and `1 = Fake`.

The notebook references [AI-image-detection-FAKE](https://huggingface.co/datasets/fcsn37/AI-image-detection-FAKE) and [AI-image-detection-REAL](https://huggingface.co/datasets/fcsn37/AI-image-detection-REAL). Dataset provenance and usage rights follow the original sources; review their terms before reuse or redistribution.

## Model Architecture

The hybrid model combines two views of each 240 × 240 RGB image: EfficientNet-B1 extracts spatial features, while an FFT plus CNN extracts frequency features. An attention gate fuses both branches before a small classifier predicts Real or Fake.

![Model architecture](assets/model_architecture.svg)

See [`diagram_code.md`](diagram_code.md) for detailed diagrams.

## Project Structure

```text
Source/
├── assets/                 # Architecture diagram
├── data/                   # Dataset, transforms, loaders, face cropping
├── engine/                 # Training, evaluation, inference
├── models/                 # Spatial, frequency, fusion, detector modules
├── notebooks/              # Data prep, training, testing, visualization
├── utils/                  # Configuration, metrics, checkpoints, plots
├── diagram_code.md         # Detailed Mermaid diagrams
├── requirements.txt
├── README.md
└── README_vi.md
```

## Installation

Use Python 3.10 or newer in a virtual environment. Run these commands from `Source/`:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install jupyter
jupyter lab
```

Open Jupyter from `Source/` so `system_path.py` can locate the project modules. Torchvision may download EfficientNet ImageNet weights on first use. The code selects CUDA when available, otherwise CPU. Install a PyTorch build compatible with your hardware if the pinned build is unsuitable.

## Usage

Run the notebooks in order as needed:

1. [`01_DataPrepare.ipynb`](notebooks/01_DataPrepare.ipynb) — inspect, preprocess, and split images. Adjust configured source paths if needed.
2. [`02_TrainingModel.ipynb`](notebooks/02_TrainingModel.ipynb) — train the four model variants. Checkpoints and logs are written to project-root `Outputs/<model-name>/`.
3. [`03_ModelTesting.ipynb`](notebooks/03_ModelTesting.ipynb) — load the best checkpoints and predict sample images. Replace hard-coded example paths for your machine.
4. [`04_VisualizeExample.ipynb`](notebooks/04_VisualizeExample.ipynb) — visualize FFT and filtered frequency maps; update the sample image path.

`FaceCrop.ipynb` is a separate exploratory Hugging Face/Google Colab workflow. `TrainingModel Old ver.ipynb` is an earlier experiment; use the numbered training notebook for the current setup.

## Results Summary

The table is transcribed from `Outputs/*/logs/evaluation_results/test_metrics.json`. It reports saved results for the same 4,000-image test split; Fake is the positive class for precision, recall, and F1. These results do not guarantee performance on other datasets.

| Model                         |   Accuracy |  Precision |     Recall |         F1 |    ROC AUC |
| ----------------------------- | ---------: | ---------: | ---------: | ---------: | ---------: |
| Spatial (EfficientNet-B1)     |     0.9793 |     0.9660 |     0.9935 |     0.9795 |     0.9975 |
| Frequency (FFT-CNN)           |     0.8800 |     0.8172 |     0.9790 |     0.8908 |     0.9562 |
| Hybrid asymmetric (512 + 256) | **0.9815** | **0.9684** | **0.9955** | **0.9818** | **0.9980** |
| Hybrid symmetric (512 + 512)  |     0.9768 |     0.9600 |     0.9950 |     0.9772 |     0.9973 |

Among these saved runs, the asymmetric hybrid has the highest accuracy, F1, and ROC AUC. Its gain over the spatial baseline is modest and specific to this split.

## Artifacts and Configuration

`Outputs/` contains the checkpoints and results shown above. Root-level `Outputs_*` directories contain other saved experiment variants. Evaluation stores metrics JSON, confusion matrices, and ROC plots in each model's `logs/evaluation_results/`. Main data paths and training defaults are defined in `utils/config.py`.

## License

This project is released for educational and research use under the terms in [`LICENSE`](LICENSE).
