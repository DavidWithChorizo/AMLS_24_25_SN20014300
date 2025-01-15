import os
import numpy as np
from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pathlib import Path
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from itertools import product
from tqdm.auto import tqdm
from timeit import default_timer as timer
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import optuna
from optuna import Trial
from optuna.samplers import TPESampler
import csv
import datetime
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import torch.nn as nn




#------------------------------------------------------------------- Task A Dataset Preparation Codes -------------------------------------------------------------------#



# 1. Set the Path to the Dataset
def get_breastmnist_path():
    """
    Get the relative path to the BreastMNIST dataset.

    This function constructs and returns the relative path to the BreastMNIST dataset
    based on the location of the current script.

    Returns:
        Path: The relative path to the BreastMNIST dataset.
    """
    # Get the directory of the current script
    script_dir = Path(__file__).parent.resolve()
    # Construct the relative path to the dataset
    breastmnist_path = script_dir.parent / 'Datasets' / 'BreastMNIST' / 'breastmnist.npz'
    
    return breastmnist_path






# 2. Load Dataset
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
        'train_images': data['train_images'],  
        'train_labels': data['train_labels'],  
        'val_images': data['val_images'],      
        'val_labels': data['val_labels'],      
        'test_images': data['test_images'],    
        'test_labels': data['test_labels']   
    }
    return dataset




#3. Apply Transformations
def get_transformations():
    """
    Get the transformations for training, validation, and test datasets.

    This function returns the transformations to be applied to the images in the
    training, validation, and test datasets. The transformations include data
    augmentation techniques for the training dataset and normalization for all datasets.

    Returns:
        tuple: A tuple containing two transformations:
            - transform_train: Transformation for the training dataset.
            - transform_val_test: Transformation for the validation and test datasets.
    """
    # Transformation for validation and test datasets
    # Converts images to tensors and normalizes them with mean and std of 0.5
    transform_val_test = transforms.Compose([
        transforms.ToTensor(),  # Convert PIL Image or numpy.ndarray to tensor
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize tensor with mean and std of 0.5
    ])

    # Transformation for training dataset
    # Includes data augmentation techniques and normalization
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
        transforms.RandomVerticalFlip(),  # Randomly flip the image vertically
        transforms.RandomRotation(30),  # Randomly rotate the image by up to 30 degrees
        transforms.ToTensor(),  # Convert PIL Image or numpy.ndarray to tensor
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize tensor with mean and std of 0.5
    ])

    return transform_train, transform_val_test

# Example usage
transform_train, transform_val_test = get_transformations()

'''#3. Apply Transformations
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
'''


#4. Provide a custom dataset class to handle the MedMNIST dataset, including loading images and labels, applying transformations, and providing the length and individual items of the dataset.
class MedMNISTDataset(Dataset):
    """
    Custom Dataset class for MedMNIST.

    This class handles the MedMNIST dataset, including loading images and labels,
    applying transformations, and providing the length and individual items of the dataset.

    Attributes:
        images (numpy.ndarray): Array of images.
        labels (numpy.ndarray): Array of labels corresponding to the images.
        transform (callable, optional): Optional transform to be applied on an image sample.
    """

    def __init__(self, images, labels, transform=None):
        """
        Initialize the MedMNISTDataset.

        Args:
            images (numpy.ndarray): Array of images.
            labels (numpy.ndarray): Array of labels corresponding to the images.
            transform (callable, optional): Optional transform to be applied on an image sample.
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Retrieve a sample from the dataset at the specified index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (image, label) where image is the transformed image and label is the corresponding label.
        """
        # Get the image and label at the specified index
        image = self.images[idx]  # Shape: (1, 28, 28)
        label = self.labels[idx]

        # Convert the image to a PIL Image
        # Squeeze removes single-dimensional entries from the shape of an array
        # Multiply by 255 to convert to the range [0, 255]
        # 'L' mode is for grayscale images
        image = Image.fromarray(np.uint8(image.squeeze() * 255), mode='L')

        # Apply the transformation if specified
        if self.transform:
            image = self.transform(image)

        return image, label

# Example usage
# Assuming images and labels are numpy arrays loaded from a dataset
# images = np.load('path_to_images.npy')
# labels = np.load('path_to_labels.npy')
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])
# dataset = MedMNISTDataset(images, labels, transform)

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




# 6. Flatten Data for Traditional ML (Optional)
def flatten_data(dataset_dict):
    '''
    Flatten the images for traditional machine learning algorithms.

    Args:
        dataset_dict (dict): Dictionary containing 'train_images', 'train_labels', 'val_images', 'val_labels', 'test_images', 'test_labels'.

    Returns:
        tuple: Tuple containing X_train, y_train, X_val, y_val, X_test, y_test.
    '''
    X_train = dataset_dict['train_images'].reshape(dataset_dict['train_images'].shape[0], -1)
    y_train = dataset_dict['train_labels']

    X_val = dataset_dict['val_images'].reshape(dataset_dict['val_images'].shape[0], -1)
    y_val = dataset_dict['val_labels']

    X_test = dataset_dict['test_images'].reshape(dataset_dict['test_images'].shape[0], -1)
    y_test = dataset_dict['test_labels']

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)



# 7. Preprocess Features for Traditional ML (Optional)
def preprocess_features(X_train, X_val, X_test, n_components=50):
    '''
    Preprocess features using StandardScaler and PCA.

    Args:
        X_train (ndarray): Training features.
        X_val (ndarray): Validation features.
        X_test (ndarray): Test features.
        n_components (int): Number of principal components.

    Returns:
        tuple: Tuple containing X_train_pca, X_val_pca, X_test_pca, scaler, pca.
    '''
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    return (X_train_pca, X_val_pca, X_test_pca), scaler, pca



# 8. Visualize Sample Data
def visualize_samples(loader, class_names, num_samples=5):
    '''
    Visualize sample images from the DataLoader.

    Args:
        loader (DataLoader): DataLoader object.
        class_names (list): List of class names.
        num_samples (int): Number of samples to visualize.

    Return: A plot of sample images.
    '''
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







#------------------------------------------------------------------- Task A Model Training Codes -------------------------------------------------------------------#

# 1. Define the CNN Model

class CNNModel_Breast(nn.Module):
    """
    CNN model for binary classification of BreastMNIST images.
    """
    def __init__(self, hidden_units=128, dropout=0.5):
        super(CNNModel_Breast, self).__init__()
        # Convolutional layer 1: Input channels=1 (grayscale), Output channels=32, Kernel size=3, Padding=1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # Max pooling layer: Reduces spatial dimensions by a factor of 2
        self.pool = nn.MaxPool2d(2, 2)
        # Convolutional layer 2: Input channels=32, Output channels=64, Kernel size=3, Padding=1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Fully connected layer 1: Input size=64*7*7 (assuming input image size 28x28), Output size=hidden_units
        self.fc1 = nn.Linear(64 * 7 * 7, hidden_units)
        # Dropout layer: Dropout rate=dropout
        self.dropout = nn.Dropout(dropout)
        # Fully connected layer 2: Output size=2 (binary classification)
        self.fc2 = nn.Linear(hidden_units, 2)

    def forward(self, x):
        """
        Forward pass of the model.
        """
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)  # Flatten the tensor
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Added on the 15th of January 2025.

class MedMNISTDataset(Dataset):
    """
    Custom Dataset class for MedMNIST.
    """
    def __init__(self, images, labels, transform=None):
        """
        Initialize the dataset with images, labels, and transformations.

        Args:
            images (numpy.ndarray): Array of images with shape (num_samples, 1, 28, 28).
            labels (numpy.ndarray): Array of labels with shape (num_samples,).
            transform (callable, optional): Transformations to apply to each image.
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Retrieves the image and label at the specified index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (transformed_image, label)
        """
        image = self.images[idx]  # Shape: (1, 28, 28)
        label = self.labels[idx]

        # Convert to PIL Image
        image = Image.fromarray(np.uint8(image.squeeze() * 255), mode='L')  # 'L' for grayscale

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, label
    


def initialize_csv_log(log_file):
    """
    Initialize the CSV log file with headers.

    Args:
        log_file (str): Path to the CSV log file.
    """
    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Timestamp', 'Model', 'Epoch', 'Train Loss', 'Val Loss', 'Train Acc', 'Val Acc'])


def append_csv_log(log_file, model_name, epoch, train_loss, val_loss, train_acc, val_acc):
    """
    Append a new entry to the CSV log file.

    Args:
        log_file (str): Path to the CSV log file.
        model_name (str): Name of the model ('CNN' or 'RandomForest').
        epoch (int): Current epoch number.
        train_loss (float): Training loss.
        val_loss (float): Validation loss.
        train_acc (float): Training accuracy.
        val_acc (float): Validation accuracy.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, model_name, epoch, train_loss, val_loss, train_acc, val_acc])


def optimize_and_train_cnn(train_loader, val_loader, device, log_file, model_save_path, n_trials=20):
    """
    Optimize hyperparameters and train the CNN model using Optuna.

    Args:
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        device (torch.device): Device to train the model on.
        log_file (str): Path to the CSV log file.
        model_save_path (str): Path to save the best model.
        n_trials (int): Number of Optuna trials for hyperparameter optimization.

    Returns:
        CNNModel: Best-trained CNN model.
        dict: Best hyperparameters found by Optuna.
    """
    initialize_csv_log(log_file)

    def objective(trial: Trial):
        # Suggest hyperparameters
        hidden_units = trial.suggest_categorical('hidden_units', [64, 128, 256])
        dropout = trial.suggest_uniform('dropout', 0.3, 0.7)
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
        num_epochs = trial.suggest_int('num_epochs', 10, 30)

        # Initialize the model
        model = CNNModel(hidden_units=hidden_units, dropout=dropout).to(device)

        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        best_val_acc = 0.0

        for epoch in range(num_epochs):
            # Training Phase
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

            train_loss = running_loss / total
            train_acc = correct / total

            # Validation Phase
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * images.size(0)
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)

            val_loss = val_loss / total
            val_acc = correct / total

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            # Logging
            append_csv_log(log_file, 'CNN', epoch+1, train_loss, val_loss, train_acc, val_acc)

            # Reporting to Optuna
            trial.report(val_acc, epoch)

            # Handle pruning based on intermediate value
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return val_acc

    # Create Optuna study
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Train the best model
    best_hidden_units = trial.params['hidden_units']
    best_dropout = trial.params['dropout']
    best_learning_rate = trial.params['learning_rate']
    best_num_epochs = trial.params['num_epochs']

    best_model = CNNModel(hidden_units=best_hidden_units, dropout=best_dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(best_model.parameters(), lr=best_learning_rate)

    best_history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc_overall = 0.0

    for epoch in range(best_num_epochs):
        # Training Phase
        best_model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = best_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # Validation Phase
        best_model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = best_model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_loss = val_loss / total
        val_acc = correct / total

        best_history["train_loss"].append(train_loss)
        best_history["val_loss"].append(val_loss)
        best_history["train_acc"].append(train_acc)
        best_history["val_acc"].append(val_acc)

        # Logging
        append_csv_log(log_file, 'CNN', epoch+1, train_loss, val_loss, train_acc, val_acc)

        # Save best model
        if val_acc > best_val_acc_overall:
            best_val_acc_overall = val_acc
            torch.save(best_model.state_dict(), model_save_path)
            print(f"Best model updated at epoch {epoch+1} with Val Acc: {val_acc:.4f}")

    print(f"Best Validation Accuracy: {best_val_acc_overall:.4f}")
    return best_model, trial.params, best_history, best_val_acc_overall