from torch.utils.data import Dataset
from os import path
import numpy as np

root_dir = 'data/processed'
test_dir = path.join(root_dir, 'test.txt')
train_dir = path.join(root_dir, 'train.txt')
val_dir = path.join(root_dir, 'val.txt')

class HAR_Dataset(Dataset):
    def __init__(self, split_type : str = 'train'):
        
        source_dir = None
        match split_type:
            case 'train':
                source_dir = train_dir
            case 'test':
                source_dir = test_dir
            case 'val':
                source_dir = val_dir
            case _:
                raise ValueError(f"Invalid split type: {split_type}. Must be one of ['train', 'test', 'val']")

        self.features = []
        self.ids = []
        self.activities = []
        self.users = [] # User IDs for potential future use (e.g., user-specific analysis or cross-validation)

        with open(source_dir, 'r') as f:
            for line in f:
                line = line.strip().split(',')

                self.features.append(np.array(line[3:], dtype=np.float32))
                self.ids.append(int(float(line[0])))
                self.activities.append(int(float(line[1])))
                self.users.append(int(float(line[2])))

    def get_class_names(self):
        return [
            'Downstairs',
            'Jogging',
            'Sitting',
            'Standing',
            'Upstairs',
            'Walking'
        ]
        
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.activities[idx], self.users[idx]
    
if __name__ == "__main__":
    dataset = HAR_Dataset(split_type='train')
    print(f"Dataset length: {len(dataset)}")
    sample_features, sample_activity, sample_user = dataset[0]
    print(f"Sample features shape: {sample_features.shape}")
    print(f"Sample activity: {sample_activity}")
    print(f"Sample user: {sample_user}")