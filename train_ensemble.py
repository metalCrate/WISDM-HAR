import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
from har_dataset import HAR_Dataset
from models import HAR_Model0, HAR_Model1
from tqdm import tqdm
from torch.backends import cudnn 
cudnn.benchmark = True  # Enable cuDNN auto-tuner for optimal performance

# Train settings
EPOCHS = 100

# Optim settings
LR = 1e-3
WEIGHT_DECAY = 1e-3

# Dataloader settings
PIN_MEMORY = True
PERSISTENT_WORKERS = True
BATCH_SIZE = 16
NUM_WORKERS = 2
SHUFFLE_TRAIN = True

class EnsembleModel:
    def __init__(self, num_models: int = 16, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.num_models = num_models
        self.models = [HAR_Model1(hidden_size=1).to(device) for _ in range(num_models)]
        self.optims = [optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY) for model in self.models]
        self.device = device

        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self, train_loader):
        losses = []
        for feat, act, user in tqdm(train_loader, desc="Training", total=len(train_loader), unit='batch'):
            feat = feat.to(self.device)
            act = act.to(self.device)
            avg_loss = self.train_batch(feat, act)
            losses.append(avg_loss)

        return sum(losses) / len(losses)
    
    def evaluate_epoch(self, val_loader):
        avg_accuracies = []
        ensemble_accuracies = []
        for feat, act, user in tqdm(val_loader, desc="Evaluating", total=len(val_loader), unit='batch'):
            feat = feat.to(self.device)
            act = act.to(self.device)
            avg_acc, ensemble_acc = self.evaluate_batch(feat, act)
            avg_accuracies.append(avg_acc)
            ensemble_accuracies.append(ensemble_acc)

        return sum(avg_accuracies) / len(avg_accuracies), sum(ensemble_accuracies) / len(ensemble_accuracies)
     
    def train_batch(self, features, activities):
        losses = []
        for model, optim in zip(self.models, self.optims):
            model.train()
            optim.zero_grad()
            outputs = model(features)
            loss = self.criterion(outputs, activities)
            losses.append(loss.item())
            loss.backward()
            optim.step()
        
        return sum(losses) / len(losses)  # Return average loss across models

    def evaluate_batch(self, features, activities):
        correct_count = 0
        total_count = 0
        all_outputs = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                outputs = model(features)
                all_outputs.append(outputs)
                _, predicted = torch.max(outputs.data, 1)
                total_count += activities.size(0)
                correct_count += (predicted == activities).sum().item()
        
        mean_outputs = torch.mean(torch.stack(all_outputs), dim=0)
        _, ensemble_predicted = torch.max(mean_outputs.data, 1)
        ensemble_correct_count = (ensemble_predicted == activities).sum().item()
        ensemble_accuracy = ensemble_correct_count / ensemble_predicted.size(0) * 100
        avg_accuracy = correct_count / (total_count) * 100
        return avg_accuracy, ensemble_accuracy

    def run_epoch(self, train_loader, val_loader):
        avg_train_loss = self.train_epoch(train_loader)
        avg_val_accuracy, ensemble_val_accuracy = self.evaluate_epoch(val_loader)
        return avg_train_loss, avg_val_accuracy, ensemble_val_accuracy
    
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    train_dataset = HAR_Dataset(split_type='train')
    val_dataset = HAR_Dataset(split_type='val')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE_TRAIN, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=PERSISTENT_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=PERSISTENT_WORKERS)

    models = EnsembleModel(num_models=8, device=device)

    for epoch in range(EPOCHS):
        avg_train_loss, avg_val_accuracy, ensemble_val_accuracy = models.run_epoch(train_loader, val_loader)

        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f}, Avg Val Accuracy: {avg_val_accuracy:.2f}%, Ensemble Val Accuracy: {ensemble_val_accuracy:.2f}%")


if __name__ == "__main__":
    train()