# Cross-Domain Sketch Recognition through Visual Transfer Learning

## Project Overview
This project implements a complete machine learning workflow for Kaggle **Drawn Apart | AICC Round 3**.

The challenge is a cross-domain image classification task:

- Train with labeled `cartoon` and `photograph` images
- Classify `sketch` images into the same classes
- Handle the fact that `sketch/unlabeled` and `sketch_test` are not labeled

According to the challenge definition:

- Submission format is CSV with two columns: `filename`, `class_name`
- Main metric is **F1 Score**
- `sketch_val` has labels via `val.csv` for validation only
- `sketch_test` is unlabeled and used for final prediction export

This repository compares:

- A ResNet34 frozen-backbone transfer-learning baseline
- A MobileNetV3 frozen baseline comparison model
- Three targeted ablations:
  - Domain-balanced Sampling
  - Class-imbalance Mitigation
  - Partial Unfreezing Fine-tune

## Repository Structure
```text
cross-domain-sketch-recognition/
|-- DATA/                          # Full dataset root (local)
|-- outputs/
|   |-- models/                    # Checkpoints and run metrics
|   `-- plots/                     # Plots and summary markdown
|-- src/
|   |-- data_pipeline.py           # Dataset discovery + Kaggle fallback
|   |-- baseline.py                # Baseline and baseline-compare training
|   |-- experiments.py             # Ablation runs and summary writing
|   `-- evaluation.py              # sketch_val/sketch_test evaluation and export
|-- environment.yml
|-- requirements.txt
|-- main.py
`-- README.md
```

## Installation and Setup
Create environment:
```bash
conda env create -f environment.yml
```

Activate environment:
```bash
conda activate sketch-recognition
```

Run pipeline:
```bash
python main.py
```

## Dataset access

Place the extracted dataset in the local DATA/ directory, or configure Kaggle CLI authentication and accept the competition rules on Kaggle before running the pipeline. The pipeline downloads the dataset if it is missing locally.

## Execution Process
When `main.py` runs, the pipeline executes these stages in order:

1. Data pipeline
- Validate local dataset structure (`cartoon`, `photograph`, `sketch/unlabeled`, `sketch_val`, `sketch_test`, `val.csv`)
- If missing, try Kaggle download fallback

2. Baseline stage
- Train/evaluate ResNet34 frozen baseline
- Reuse cached checkpoint/metrics when signatures match

3. Baseline comparison stage
- Run MobileNetV3 frozen baseline comparison
- Report delta versus ResNet34 baseline

4. Ablation stage
- Run three ablations
- Report weighted F1 and delta versus baseline

5. Evaluation stage
- Evaluate on `sketch_val` (metric + plots)
- Infer on `sketch_test` and export prediction CSV

## Required Packages
Core dependencies:

- kaggle
- numpy
- matplotlib
- Pillow
- pandas
- scikit-learn
- torch
- torchvision
- tqdm

## Third-Party Code Disclosure
This project uses public third-party resources and libraries:

- Kaggle competition dataset: Drawn Apart | AICC Round 3
- `torchvision` pretrained models (ResNet34, MobileNetV3)
- Standard Python ML/CV libraries listed in `requirements.txt`
