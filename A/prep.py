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




import os
import numpy as np
from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Define Relative Paths

# Get the directory of the current script
script_dir = Path(__file__).parent.resolve()
# Construct the relative path to the dataset
bloodmnist_path = script_dir.parent / 'Datasets' / 'BloodMNIST' / 'bloodmnist.npz'
breastmnist_path = script_dir.parent / 'Datasets' / 'BreastMNIST' / 'breastmnist.npz'

# 2. Load the .npz Files
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
breastmnist_data = load_npz_dataset(breastmnist_path)
bloodmnist_data = load_npz_dataset(bloodmnist_path)

# visually check if these datasets are loaded correctly
print(breastmnist_data.keys())
print(breastmnist_data['train_images'].shape)

# 3. Define Transformations
transform_val_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 4. Create Custom Dataset Class
class MedMNISTDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]  # Shape: (1, 28, 28)
        label = self.labels[idx]

        # Convert to PIL Image
        image = Image.fromarray(np.uint8(image.squeeze() * 255), mode='L')  # 'L' mode for grayscale

        if self.transform:
            image = self.transform(image)

        return image, label

# 5. Create DataLoaders
def create_dataloaders(dataset_dict, batch_size=32):
    train_dataset = MedMNISTDataset(
        images=dataset_dict['train_images'],
        labels=dataset_dict['train_labels'],
        transform=transform_train
    )

    val_dataset = MedMNISTDataset(
        images=dataset_dict['val_images'],
        labels=dataset_dict['val_labels'],
        transform=transform_val_test
    )

    test_dataset = MedMNISTDataset(
        images=dataset_dict['test_images'],
        labels=dataset_dict['test_labels'],
        transform=transform_val_test
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader

# Create DataLoaders for BloodMNIST
blood_train_loader, blood_val_loader, blood_test_loader = create_dataloaders(bloodmnist_data, batch_size=32)

# Create DataLoaders for BreastMNIST
breast_train_loader, breast_val_loader, breast_test_loader = create_dataloaders(breastmnist_data, batch_size=32)

# 6. Flatten Data for Traditional ML (Optional)
def flatten_data(dataset_dict):
    X_train = dataset_dict['train_images'].reshape(dataset_dict['train_images'].shape[0], -1)
    y_train = dataset_dict['train_labels']

    X_val = dataset_dict['val_images'].reshape(dataset_dict['val_images'].shape[0], -1)
    y_val = dataset_dict['val_labels']

    X_test = dataset_dict['test_images'].reshape(dataset_dict['test_images'].shape[0], -1)
    y_test = dataset_dict['test_labels']

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

(X_b_train, y_b_train), (X_b_val, y_b_val), (X_b_test, y_b_test) = flatten_data(breastmnist_data)

# 7. Preprocess Features for Traditional ML (Optional)
def preprocess_features(X_train, X_val, X_test, n_components=50):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    return (X_train_pca, X_val_pca, X_test_pca), scaler, pca

transformed_data, scaler, pca = preprocess_features(X_b_train, X_b_val, X_b_test, n_components=50)
X_b_train_pca, X_b_val_pca, X_b_test_pca = transformed_data

# 8. Visualize Sample Data
def visualize_samples(loader, class_names, num_samples=5):
    dataiter = iter(loader)
    images, labels = dataiter.next()
    images = images.numpy()

    fig, axes = plt.subplots(1, num_samples, figsize=(15,3))
    for idx in range(num_samples):
        ax = axes[idx]
        img = images[idx].squeeze()
        img = (img * 0.5) + 0.5  # Denormalize to [0,1]
        ax.imshow(img, cmap='gray')
        ax.set_title(class_names[labels[idx]])
        ax.axis('off')
    plt.show()
