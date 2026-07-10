import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
from har_dataset import HAR_Dataset
from models import HAR_Model0, HAR_Model1, HAR_ModelDeepAdjst
from tqdm import tqdm
from torch.backends import cudnn 
cudnn.benchmark = True  # Enable cuDNN auto-tuner for optimal performance

# Train settings
EPOCHS = 2000

# Optim settings
LR = 1e-3
WEIGHT_DECAY = 1e-5
LABEL_SMOOTHING = 0.175

# Dataloader settings
PIN_MEMORY = True
PERSISTENT_WORKERS = True
BATCH_SIZE = 128
NUM_WORKERS = 4
SHUFFLE_TRAIN = True

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Weights as calculated from preprocess_data.py
    class_weights = torch.tensor([0.43392598581926484, 0.555692289164672, 1.4287973458667087, 1.7102270972346512, 2.950979869659838, 3.6707308988609753], dtype=torch.float32)
    class_weight_lerp = 0.33  # Example value, replace with actual hyperparameter
    class_weights = class_weight_lerp * class_weights + (1 - class_weight_lerp) * torch.ones_like(class_weights)
    
    class_weights = class_weights.to(device)


    train_dataset = HAR_Dataset(split_type='train')
    val_dataset = HAR_Dataset(split_type='val')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE_TRAIN, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=PERSISTENT_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=PERSISTENT_WORKERS)

    model = HAR_ModelDeepAdjst(hidden_size=256,
                               depth=4,
                               norm='Identity',
                               activation='LeakyReLU',
                               dropout=0.7,
                               residual=True
                               ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, steps_per_epoch=len(train_loader), epochs=EPOCHS)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING).to(device)

    for epoch in range(EPOCHS):
        
        tr_losses = []
        val_losses = []

        model.train()
        for features, activities, users in train_loader: #tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", total=len(train_loader), unit='batch'):
            features = features.to(device)
            activities = activities.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, activities)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            tr_losses.append(loss.item())
        
        correct_count = 0
        total_count = 0
        model.eval()
        for features, activities, users in val_loader: #tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{EPOCHS}", total=len(val_loader), unit='batch'):
            features = features.to(device)
            activities = activities.to(device)

            with torch.no_grad():
                outputs = model(features)
                loss = criterion(outputs, activities)
                val_losses.append(loss.item())
                total_count += activities.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct_count += (predicted == activities).sum().item()
        
        accuracy = correct_count / total_count * 100
        print(f"Epoch {epoch+1}/{EPOCHS}, Tr Loss: {sum(tr_losses)/len(tr_losses):.4f}, Val Loss: {sum(val_losses)/len(val_losses):.4f}")
        print(f"Validation Accuracy: {accuracy:.2f}%")

    torch.save(model.state_dict(), "har_model_deep_final.pth") 
if __name__ == "__main__":
    train()