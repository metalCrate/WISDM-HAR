# Dataset loading stays local so training and evaluation share the same split format.
from torch.utils.data import Dataset
from os import path
import numpy as np

# Centralize split paths so file handling stays consistent across consumers.
root_dir = 'data/processed'
test_dir = path.join(root_dir, 'test.txt')
train_dir = path.join(root_dir, 'train.txt')
val_dir = path.join(root_dir, 'val.txt')

class HAR_Dataset(Dataset):
    """Dataset wrapper for the preprocessed WISDM HAR text splits.

    Parameters
    ----------
    split_type : str, optional
        Which split to load, by default 'train'.
    """
    def __init__(self, split_type : str = 'train'):
        # Resolve the requested split once so the rest of the loader can stay simple.
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

            # Load eagerly because the processed splits are small enough for in-memory use.
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
        """Return the canonical class labels used by the processed dataset.

        Returns
        -------
        list[str]
            Ordered activity names aligned with the encoded labels.
        """
        return [
            'Downstairs',
            'Jogging',
            'Sitting',
            'Standing',
            'Upstairs',
            'Walking'
        ]
        
    def __len__(self):
        """Return the number of samples loaded into memory.

        Returns
        -------
        int
            Number of dataset examples.
        """
        return len(self.features)

    def __getitem__(self, idx):
        """Fetch one sample, its activity label, and its user id.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        tuple
            A tuple of (features, activity, user).
        """
        return self.features[idx], self.activities[idx], self.users[idx]
    
if __name__ == "__main__":
    # Keep the module self-check lightweight so dataset parsing can be verified quickly.
    dataset = HAR_Dataset(split_type='train')
    print(f"Dataset length: {len(dataset)}")
    sample_features, sample_activity, sample_user = dataset[0]
    print(f"Sample features shape: {sample_features.shape}")
    print(f"Sample activity: {sample_activity}")
    print(f"Sample user: {sample_user}")