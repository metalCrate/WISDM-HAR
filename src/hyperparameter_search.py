# Hyperparameter search is isolated so experiments do not affect the training script.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
from src.har_dataset import HAR_Dataset
from src.models import HAR_Model0, HAR_Model1, HAR_ModelDeepAdjst
from tqdm import tqdm
from torch.backends import cudnn 
import optuna

# Keep the trial budget explicit so runs remain comparable.
EPOCHS = 50

cudnn.benchmark = True


# Keep data-loader behavior fixed so the search space stays focused on model quality.
PIN_MEMORY = True
PERSISTENT_WORKERS = True

NUM_WORKERS = 4
SHUFFLE_TRAIN = True

def objective(trial: optuna.trial.Trial):
    """Evaluate one Optuna trial on the validation split.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Active Optuna trial used to sample hyperparameters.

    Returns
    -------
    float
        Validation accuracy for the sampled configuration.
    """
    # Select the runtime device once so the trial can move tensors consistently.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Hyperparameter search space
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    WEIGHT_DECAY = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)
    LABEL_SMOOTHING = trial.suggest_float("label_smoothing", 0.01, 0.5)
    BATCH_SIZE = 256 # frozen due to performance issues (lower is generally faster converging but slower to finish an epoch) # trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256])
    norm_fn = trial.suggest_categorical("norm_fn", ["LayerNorm", "BatchNorm1d", "Identity"])
    activation_fn = trial.suggest_categorical("activation_fn", ["ReLU", "GELU", "SiLU", "Tanh", 'LeakyReLU'])
    class_weight_lerp = trial.suggest_float("class_weight_lerp", 0.0, 1.0)
    scheduler = trial.suggest_categorical("scheduler", ["OneCycleLR", "CosineAnnealingLR", "None"])
    
    # Weights as calculated from preprocess_data.py
    class_weights = torch.tensor([0.43392598581926484, 0.555692289164672, 1.4287973458667087, 1.7102270972346512, 2.950979869659838, 3.6707308988609753], dtype=torch.float32)

    class_weights = class_weight_lerp * class_weights + (1 - class_weight_lerp) * torch.ones_like(class_weights)
    class_weights = class_weights.to(device)

    train_dataset = HAR_Dataset(split_type='train')
    val_dataset = HAR_Dataset(split_type='val')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE_TRAIN, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=PERSISTENT_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=PERSISTENT_WORKERS)

    model = HAR_ModelDeepAdjst(hidden_size=32,
                               norm=norm_fn,
                               activation=activation_fn
                               ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    match scheduler:
        case "OneCycleLR":
            scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=EPOCHS)
        case "CosineAnnealingLR":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        case "None":
            scheduler = None

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING).to(device)

    # Optimize against validation accuracy because the search goal is generalization.
    for epoch in range(EPOCHS):
        
        tr_losses = []
        val_losses = []

        model.train()
        for features, activities, users in train_loader:
            features = features.to(device)
            activities = activities.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, activities)
            loss.backward()
            optimizer.step()
            if(scheduler == 'OneCycleLR'):
                scheduler.step()
            optimizer.zero_grad()
            tr_losses.append(loss.item())
        
        correct_count = 0
        total_count = 0
        model.eval()
        for features, activities, users in val_loader:
            features = features.to(device)
            activities = activities.to(device)

            with torch.no_grad():
                outputs = model(features)
                loss = criterion(outputs, activities)
                val_losses.append(loss.item())
                total_count += activities.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct_count += (predicted == activities).sum().item()
        
        if(scheduler == 'CosineAnnealingLR'):
            scheduler.step()

        accuracy = correct_count / total_count * 100
        trial.report(accuracy, epoch)

        if trial.should_prune():
            print(f"Pruned at validation Accuracy: {accuracy:.2f}%")
            raise optuna.TrialPruned()
        
        return accuracy


if __name__ == "__main__":
    # Run the study directly so experiments can be launched without extra code.
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50) 

    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")