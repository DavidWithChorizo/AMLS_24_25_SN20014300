import numpy as np
from pathlib import Path

def load_npz_dataset(npz_path):
    """
    Load dataset from an .npz file.

    Args:
        npz_path (str): Path to the .npz file.

    Returns:
        dict: Dictionary containing 'train_images', 'train_labels', 'val_images', 'val_labels', 'test_images', 'test_labels'.
    """
    data = np.load(npz_path)
    dataset = {
        'train_images': data['train_images'],  # Shape: (546, 1, 28, 28)
        'train_labels': data['train_labels'],  # Shape: (546,)
        'val_images': data['val_images'],      # Shape: (78, 1, 28, 28)
        'val_labels': data['val_labels'],      # Shape: (78,)
        'test_images': data['test_images'],    # Shape: (156, 1, 28, 28)
        'test_labels': data['test_labels']     # Shape: (156,)
    }
    return dataset

# Example usage:
#breastmnist_data = load_npz_dataset('')

# Get the directory of the current script
script_dir = Path(__file__).parent.resolve()

# Construct the relative path to the dataset
bloodmnist_path = script_dir.parent / 'Datasets' / 'BloodMNIST' / 'bloodmnist.npz'

print(bloodmnist_path)