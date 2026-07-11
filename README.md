# Human Activity Recognition using Smartphone Sensors (WISDM) - Small MLP
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview
This project implements an end-to-end pipeline for Human Activity Recognition (HAR) using the WISDM dataset. It includes data preprocessing and splitting, exploratory analysis, hyperparameter tuning (via Optuna), and training of a lightweight Multi-Layer Perceptron (MLP) achieving **93.81% test accuracy** with a 73K parameter model.

## Key Features
- Handles sparse ARFF format parsing without external Weka dependencies.
- Implements stratified train/validation/test splits to preserve class distribution.
- Supports class-weighting to handle severe data imbalance (Class 5 is only 4.5% of the entire dataset).
- Hyperparameter tuning with Optuna.
- Reproducible training with configurable YAML files.

## Results
| Metric | Value |
| :--- | :--- |
| **Test Accuracy** | **93.81%** |
| **Macro F1-Score** | 89.7% |
| **Weighted F1-Score** | 92.4% |
| **Model Parameters** | 72,454  (~0.3 MB) |

![Normalized Confusion Matrix](plots/confusion_matrix.png)

> **Note:** While state-of-the-art (SOTA) on WISDM reaches ~99%, this project focuses on delivering a clean, reproducible, and lightweight baseline suitable for resource-constrained devices, serving as a strong foundation for further experimentation.

## Installation & Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/human-activity-recognition.git
   cd human-activity-recognition
2. pip install -r requirements.txt
3. python src/preprocessing.py
4. python train.py
5. python evaluate.py

## Citation

If you use this dataset in your research or project, please cite the following paper:

> Kwapisz, J. R., Weiss, G. M., & Moore, S. A. (2010). Activity Recognition using Cell Phone Accelerometers. In *Proceedings of the Fourth International Workshop on Knowledge Discovery from Sensor Data (at KDD-10)*. Washington DC.

**Dataset:** WISDM Activity Prediction Dataset (v1.1), Wireless Sensor Data Mining (WISDM) Lab, Fordham University.