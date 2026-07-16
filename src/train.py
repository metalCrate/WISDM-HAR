# Training keeps the full experiment pipeline in one place for repeatable runs.
from unittest import case

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
from src import HAR_Dataset, HAR_ModelDeepAdjst
import yaml
import argparse
from tqdm import tqdm
from torch.backends import cudnn 
import matplotlib.pyplot as plt
from os import path
cudnn.benchmark = True  # Enable cuDNN auto-tuner for optimal performance during training.


def create_plot(train_losses, validation_losses, validation_accuracies, save_name="training_plot.png"):
    """Plot training curves for loss and validation accuracy.

    Parameters
    ----------
    train_losses : list[float]
        Per-epoch training losses.
    validation_losses : list[float]
        Per-epoch validation losses.
    validation_accuracies : list[float]
        Per-epoch validation accuracies.
    save_name : str, optional
        Plot filename, by default "training_plot.png".

    Side Effects
    ------------
    Writes a PNG plot into the plots directory.
    """
    # Plotting after training keeps the main loop free of visualization overhead.
    save_path = path.join("plots", save_name)
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, validation_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, validation_accuracies, label='Validation Accuracy', color='orange')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    #plt.show()


def train_from_config(config_path :str = 'configs/config.yaml'):
    """Train the configurable HAR model from a YAML experiment config.

    Parameters
    ----------
    config_path : str, optional
        Path to the YAML configuration file, by default 'configs/config.yaml'.

    """
    # ------- 1. Load Config -------
    # Load config first so every downstream choice follows one source of truth.
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    num_epochs = config['train']['epochs']

    torch.manual_seed(config['random_seed'])

    scheduler_type = config['train']['scheduler']  # Options: "OneCycleLR", "CosineAnnealingLR", "None"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Weights as calculated from preprocess_data.py
    # class_weight_lerp is used for compatibility with hyperparameter search. 
    class_weights = torch.tensor(config['train']['class_weights'], dtype=torch.float32)
    class_weight_lerp = config['train']['class_weight_lerp'] # set to 1 to disable class weighting
    class_weights = class_weight_lerp * class_weights + (1 - class_weight_lerp) * torch.ones_like(class_weights)
    class_weights = class_weights.to(device)


    # Recreate the loaders for the active config so batch behavior stays reproducible.
    train_dataset = HAR_Dataset(split_type='train')
    val_dataset = HAR_Dataset(split_type='val')

    train_loader = DataLoader(train_dataset,
                              batch_size=config['data']['batch_size'],
                              shuffle=config['data']['shuffle_train'],
                              num_workers=config['data']['num_workers'],
                              pin_memory=config['data']['pin_memory'],
                              persistent_workers=config['data']['persistent_workers']
                              )
    
    val_loader = DataLoader(val_dataset,
                            batch_size=config['data']['batch_size'],
                            shuffle=False,
                            num_workers=config['data']['num_workers'],
                            pin_memory=config['data']['pin_memory'],
                            persistent_workers=config['data']['persistent_workers']
                            )

    model = HAR_ModelDeepAdjst(
        input_size=config['model']['input_size'],
        hidden_size=config['model']['hidden_size'],
        depth=config['model']['depth'],
        norm=config['model']['norm'],
        activation=config['model']['activation'],
        dropout=config['model']['dropout'],
        residual=config['model']['residual']
                               ).to(device)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['train']['lr'],
        weight_decay=config['train']['weight_decay']
    )

    match config['train']['scheduler']:
        case "OneCycleLR":
            final_div_factor = config['train']['lr'] / config['train']['end_lr']
            scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config['train']['lr'], steps_per_epoch=len(train_loader), epochs=num_epochs, final_div_factor=final_div_factor)
        case "CosineAnnealingLR":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=config['train']['end_lr'])
        case "CosineAnnealingWarmRestarts":
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=config['train']['T_0'],
                T_mult=config['train']['T_mult'],
                eta_min=config['train']['end_lr']
            )
        case "None":
            scheduler = None

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=config['train']['label_smoothing']).to(device)

    # Track these metrics separately so the saved plot reflects epoch-level trends.
    train_losses = []
    validation_losses = []
    validation_accuracies = []

    # Keep the optimization loop straightforward so schedule and logging stay easy to audit.
    for epoch in range(num_epochs):
        
        _tr_losses = []
        _val_losses = []

        model.train()
        for features, activities, users in train_loader:
            features = features.to(device)
            activities = activities.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, activities)
            loss.backward()
            optimizer.step()

            if scheduler_type == 'OneCycleLR':
                scheduler.step()

            optimizer.zero_grad()
            _tr_losses.append(loss.item())
        

        correct_count = 0
        total_count = 0
        model.eval()
        for features, activities, users in val_loader:
            features = features.to(device)
            activities = activities.to(device)

            with torch.no_grad():
                outputs = model(features)
                loss = criterion(outputs, activities)
                _val_losses.append(loss.item())
                total_count += activities.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct_count += (predicted == activities).sum().item()
        
        if scheduler_type in ['CosineAnnealingLR', 'CosineAnnealingWarmRestarts']:
            scheduler.step()

        accuracy = correct_count / total_count * 100
        _tr_loss = sum(_tr_losses)/len(_tr_losses)
        _val_loss = sum(_val_losses)/len(_val_losses)

        train_losses.append(_tr_loss)
        validation_losses.append(_val_loss)
        validation_accuracies.append(accuracy)

        print(f"Epoch {epoch+1}/{num_epochs}, Tr Loss: {_tr_loss:.4f}, Val Loss: {_val_loss:.4f}")
        print(f"Validation Accuracy: {accuracy:.2f}%")  
    
    # Save artifacts only after the final validation pass confirms the model state.
    print("==== Training Summary ====")
    print(f"Final validation accuracy: {validation_accuracies[-1]:.2f}%")
    print(f"Model size: {sum(p.numel() for p in model.parameters())} parameters")
    
    torch.save(model.state_dict(), "models/har_model_deep_final.pth")
    create_plot(train_losses, validation_losses, validation_accuracies, save_name=f"training_plot_{num_epochs}epochs.png")

    print(f"Training complete. Model saved as 'har_model_deep_final.pth' and training plot saved as 'training_plot_{num_epochs}epochs.png'.")

if __name__ == "__main__":
    # Keep the script runnable directly while also exposing the function for imports.
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to the configuration file')
    args = parser.parse_args()
    train_from_config(args.config)