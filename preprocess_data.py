import pandas as pd
from scipy.io import arff
import re
from os import path
import io
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split

def generate_split_data():
    data_root = 'original_data'
    data_path = path.join(data_root, 'WISDM_ar_v1.1','WISDM_ar_v1.1_transformed.arff')

    out_root = 'split_data'

    out_train = path.join(out_root, 'train.txt')
    out_val = path.join(out_root, 'val.txt')
    out_test = path.join(out_root, 'test.txt')

    train_ratio = 0.6
    val_to_test_ratio = 0.5

    classes = ["Walking" , "Jogging" , "Upstairs" , "Downstairs" , "Sitting" , "Standing"]
    class_to_idx = {c: str(i) for i, c in enumerate(classes)}


    stripped_values = []

    class_counts = {c: 0 for c in classes}

    print("Processing and splitting data.\n")
    with open(data_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments, and attribute definitions
            if not line:
                continue
            if line.startswith('@'):
                continue
            
            values = line.split(',')
            values = [v.strip() for v in values]
            class_counts[values[-1]] += 1
            values[-1] = class_to_idx[values[-1]]  # Convert class label to index
            values.insert(1, values.pop()) # Transform into [id, class, user, ...]
            values = [float(v.replace('?', '0')) for v in values] # Ensure float conversion for all values
            stripped_values.append(values)


    print(f"Total samples found: {len(stripped_values)}")
    #print("Last sample:", stripped_values[-1]) 
    print("Class counts:")
    for class_name, count in class_counts.items(): 
        print(f"  {class_name}: {count}")

    print("Class occurence ratios:")
    _occurences = [count / len(stripped_values) for count in class_counts.values()]
    print(_occurences)

    print("Suggested criterion weights:")
    _weights = [1.0 / (occur + 1e-8) / 6 for occur in _occurences]
    print(_weights)

    mean_values = np.mean(stripped_values, axis=0)
    std_values = np.std(stripped_values, axis=0)

    normalized_values = (stripped_values - mean_values) / (std_values + 1e-8)
    final_data = stripped_values

    # Normalize fields (except id, class and user) to have mean 0 and std 1
    for i in range(len(final_data)):
        point = final_data[i]
        
        for j in range(3,len(point)):
            point[j] = normalized_values[i][j]

    # Generate train, validation, and test splits
    # Note: each data point is in the format [id, class, user, ...features]

    final_data = np.array(final_data)
    labels = [row[1] for row in final_data]  # Extract labels for stratification

    train_rows, temp_rows, _, temp_labels = train_test_split(
        final_data, labels,
        train_size=train_ratio,
        stratify=labels,
        random_state=11
    )

    val_rows, test_rows, _, _ = train_test_split(
    temp_rows, temp_labels,
    train_size = val_to_test_ratio,
    stratify=temp_labels,
    random_state=11 # For reproducibility
    )

    print(f"Train samples: {len(train_rows)}")
    print(f"Validation samples: {len(val_rows)}")
    print(f"Test samples: {len(test_rows)}")


    # Save the splits to text files
    np.savetxt(out_train, train_rows, fmt='%s', delimiter=',')
    np.savetxt(out_val, val_rows, fmt='%s', delimiter=',')
    np.savetxt(out_test, test_rows, fmt='%s', delimiter=',')

    print(f"Data splits saved to {out_root} directory.")

if __name__ == "__main__":
    generate_split_data()