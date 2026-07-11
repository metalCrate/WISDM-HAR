import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
from src.har_dataset import HAR_Dataset
from src.models import HAR_Model0, HAR_Model1, HAR_ModelDeepAdjst
from tqdm import tqdm
from torch.backends import cudnn 
import yaml
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from os import path
import csv

def plot_confusion_matrix(cm, class_names, save_name="confusion_matrix.png", normalized = True):
    save_path = path.join("plots", save_name)

    if normalized:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_normalized = cm.astype('float') / (row_sums + 1e-9)
        fmt = '.2f'
        title = 'Normalized Confusion Matrix (Proportions)'
    else:
        cm_normalized = cm
        fmt = 'd'
        title = 'Confusion Matrix (Counts)'

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontsize=16)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)  # Save as high-res image
    plt.show()

def test_model(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = config['data']['batch_size']
    model_state_path = 'har_model_deep_final.pth'

    model = HAR_ModelDeepAdjst(
        input_size=config['model']['input_size'],
        hidden_size=config['model']['hidden_size'],
        output_size=config['model']['output_size'],
        depth=config['model']['depth'],
        norm=config['model']['norm'],
        activation=config['model']['activation'],
        dropout=config['model']['dropout'],
        residual=config['model']['residual']
                                ).to(device)

    # Load the trained model weights
    model.load_state_dict(torch.load(model_state_path, map_location=device))

    test_dataset = HAR_Dataset(split_type='test')
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    accuracies = []
    all_predictions = []
    all_labels = []

    model.eval()
    for feat, act, user in tqdm(test_loader, desc="Testing", total=len(test_loader), unit='batch'):
        feat = feat.to(device)
        act = act.to(device)
        with torch.no_grad():
            outputs = model(feat)
            _, predicted = torch.max(outputs.data, 1)
            
            # For scoring
            accuracy = (predicted == act).sum().item() / act.size(0) * 100
            accuracies.append(accuracy)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(act.cpu().numpy())

    print(f"Average Accuracy: {np.mean(accuracies):.2f}%")

    print(f"Model size: {sum(p.numel() for p in model.parameters())} parameters")
    # print model memory size
    print(f"Model memory size: {sum(p.element_size() * p.numel() for p in model.parameters()) / (1024 ** 2):.2f} MB")

    y_pred = np.array(all_predictions)
    y_true = np.array(all_labels)

    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    print(f"F1 Score (Macro): {f1_macro:.4f}")
    print(f"F1 Score (Weighted): {f1_weighted:.4f}")

    class_names = test_dataset.get_class_names()
    print("\nClassification Report:")

    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)
    plot_confusion_matrix(cm, class_names, save_name="confusion_matrix.png")

if __name__ == "__main__":
    test_model(config_path='configs/config.yaml')