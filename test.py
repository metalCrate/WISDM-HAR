import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
from har_dataset import HAR_Dataset
from models import HAR_Model0, HAR_Model1, HAR_ModelDeepAdjst
from tqdm import tqdm
from torch.backends import cudnn 

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 64
    model_state_path = 'har_model_deep_final.pth'

    model = HAR_ModelDeepAdjst(hidden_size=256,
                                depth=4,
                                norm='Identity',
                                activation='LeakyReLU',
                                dropout=0.65,
                                residual=True
                                ).to(device)

    # Load the trained model weights
    model.load_state_dict(torch.load(model_state_path, map_location=device))

    test_dataset = HAR_Dataset(split_type='test')
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    accuracies = []
    model.eval()
    for feat, act, user in tqdm(test_loader, desc="Testing", total=len(test_loader), unit='batch'):
        feat = feat.to(device)
        act = act.to(device)
        with torch.no_grad():
            outputs = model(feat)
            _, predicted = torch.max(outputs.data, 1)
            total_count = act.size(0)
            correct_count = (predicted == act).sum().item()
            accuracy = correct_count / total_count * 100
            accuracies.append(accuracy)

    average_accuracy = sum(accuracies) / len(accuracies)
    print(f"Average Test Accuracy: {average_accuracy:.2f}%")

    print(f"Model size: {sum(p.numel() for p in model.parameters())} parameters")