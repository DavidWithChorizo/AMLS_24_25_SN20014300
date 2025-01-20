import os
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from itertools import product
from tqdm.auto import tqdm
from timeit import default_timer as timer
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import optuna
from optuna import Trial
from optuna.samplers import TPESampler
import csv
import datetime
from medmnist import BreastMNIST, BloodMNIST

#------------------------------------------------------------------- Dataset Preparation Codes -------------------------------------------------------------------#


#--------------------------------- Task Agnostic Codes ---------------------------------#


#Calculate Mean and Standard Deviation for Normalization
def calculate_mean_std(dataset_loader):
    """
    Calculate the mean and standard deviation for normalization based on the dataset.

    Args:
        dataset_loader (DataLoader): DataLoader for the dataset to compute statistics.

    Returns:
        tuple: (mean, std) where both are lists containing mean and standard deviation per channel.
    """
    mean = 0.0
    std = 0.0
    total_samples = 0

    # Compute mean and std for each channel
    for inputs, _ in dataset_loader:
        batch_samples = inputs.size(0)
        mean += inputs.mean([0, 2, 3]) * batch_samples
        std += inputs.std([0, 2, 3]) * batch_samples
        total_samples += batch_samples

    mean /= total_samples
    std /= total_samples
    return mean.tolist(), std.tolist()








#CNN Transformation Pipeline (With Augmentation)
def get_transforms_for_cnn(mean: list, std: list):
    """
    Define transformation pipelines for CNN (training + validation/testing).

    Args:
        mean (list): Mean values for normalization.
        std (list): Standard deviation values for normalization.

    Returns:
        tuple: (transform_train, transform_val_test)
            where transform_train includes data augmentation,
            and transform_val_test includes only normalization.
    """
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    transform_val_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return transform_train, transform_val_test






#Random Forest Transformation Pipeline (Without Augmentation)
def get_transforms_for_rf(mean: list, std: list):
    """
    Define transformation pipelines for Random Forest (training + validation/testing).

    Args:
        mean (list): Mean values for normalization.
        std (list): Standard deviation values for normalization.

    Returns:
        tuple: (transform_train, transform_val_test)
            where transform_train includes only normalization,
            and transform_val_test includes only normalization.
    """
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    transform_val_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return transform_train, transform_val_test









#Flattening Features for Random Forest
def flatten_features(data_loader: DataLoader):
    """
    Flatten all batches of data in a DataLoader for use in Random Forest.

    Args:
        data_loader (DataLoader): PyTorch DataLoader object.

    Returns:
        tuple: (X, y) where X is a 2D numpy array of flattened features, 
               and y is a 1D numpy array of labels.
    """
    X, y = [], []
    for inputs, labels in data_loader:
        # Flatten [batch_size, channels, height, width] -> [batch_size, channels*height*width]
        flattened = inputs.view(inputs.size(0), -1).numpy()
        X.append(flattened)
        y.append(labels.numpy())
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    return X, y








#Data Preprocessing for Random Forest (Standardization + PCA)
def preprocess_for_rf(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray, n_components=50):
    """
    Standardize the data and reduce dimensionality using PCA for Random Forest.

    Args:
        X_train (numpy.ndarray): Training features.
        X_val (numpy.ndarray): Validation features.
        X_test (numpy.ndarray): Test features.
        n_components (int): Number of PCA components.

    Returns:
        tuple: ((X_train_pca, X_val_pca, X_test_pca), scaler, pca)
    """
    # Standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # PCA
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    return (X_train_pca, X_val_pca, X_test_pca), scaler, pca











#--------------------------------- Task Specific Codes: Task A (Breast) ---------------------------------#

#Load BreastMNIST Datasets with Corresponding Transformations
def load_breastmnist_datasets(
    batch_size: int,
    download: bool,
    data_dir: str,
    transform_train_cnn,
    transform_val_test_cnn,
    transform_train_rf,
    transform_val_test_rf
):
    """
    Load BreastMNIST datasets for both CNN and Random Forest with corresponding transformations.

    Args:
        batch_size (int): Number of samples per batch.
        download (bool): Whether to download the dataset if not present.
        data_dir (str): Directory where the dataset is stored or will be downloaded to.
        transform_train_cnn (transforms.Compose): Training transform for CNN.
        transform_val_test_cnn (transforms.Compose): Val/test transform for CNN.
        transform_train_rf (transforms.Compose): Training transform for RF.
        transform_val_test_rf (transforms.Compose): Val/test transform for RF.

    Returns:
        tuple: 
            - (cnn_train_loader, cnn_val_loader, cnn_test_loader)
            - (rf_train_loader, rf_val_loader, rf_test_loader)
    """
    # Load datasets for CNN
    train_dataset_cnn = BreastMNIST(split='train', transform=transform_train_cnn, download=download, root=data_dir)
    val_dataset_cnn   = BreastMNIST(split='val',   transform=transform_val_test_cnn, download=download, root=data_dir)
    test_dataset_cnn  = BreastMNIST(split='test',  transform=transform_val_test_cnn, download=download, root=data_dir)

    # Create DataLoaders for CNN
    cnn_train_loader = DataLoader(train_dataset_cnn, batch_size=batch_size, shuffle=True)
    cnn_val_loader   = DataLoader(val_dataset_cnn,   batch_size=batch_size, shuffle=False)
    cnn_test_loader  = DataLoader(test_dataset_cnn,  batch_size=batch_size, shuffle=False)

    # Load datasets for Random Forest
    train_dataset_rf = BreastMNIST(split='train', transform=transform_train_rf, download=download, root=data_dir)
    val_dataset_rf   = BreastMNIST(split='val',   transform=transform_val_test_rf, download=download, root=data_dir)
    test_dataset_rf  = BreastMNIST(split='test',  transform=transform_val_test_rf, download=download, root=data_dir)

    # Create DataLoaders for Random Forest (no shuffle needed)
    rf_train_loader = DataLoader(train_dataset_rf, batch_size=batch_size, shuffle=False)
    rf_val_loader   = DataLoader(val_dataset_rf,   batch_size=batch_size, shuffle=False)
    rf_test_loader  = DataLoader(test_dataset_rf,  batch_size=batch_size, shuffle=False)

    return (
        (cnn_train_loader, cnn_val_loader, cnn_test_loader),
        (rf_train_loader, rf_val_loader, rf_test_loader)
    )












#Start Data Preparation for BreastMNIST
def prepare_breastmnist_data(batch_size=32, download=True, data_dir="data_breast", n_components=50):
    """
    Prepare BreastMNIST dataset for both CNN and Random Forest.

    Steps:
    1. Compute mean/std for normalization (on training data).
    2. Define transform pipelines for CNN and RF.
    3. Load datasets and create DataLoaders for CNN and RF.
    4. Flatten features for RF.
    5. Standardize and apply PCA for RF.

    Args:
        batch_size (int, optional): Batch size. Defaults to 32.
        download (bool, optional): Download dataset if not present. Defaults to True.
        data_dir (str, optional): Directory for storing/downloading data. Defaults to "data_breast".
        n_components (int, optional): PCA components for Random Forest. Defaults to 50.

    Returns:
        tuple: 
            - cnn_train_loader, cnn_val_loader, cnn_test_loader
            - (X_train_rf_pca, y_train_rf), (X_val_rf_pca, y_val_rf), (X_test_rf_pca, y_test_rf)
            - scaler (StandardScaler), pca (PCA)
    """
    os.makedirs(data_dir, exist_ok=True)

    # Step 1: Compute mean/std from initial transform (no augmentation)
    temp_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    temp_dataset   = BreastMNIST(split='train', transform=temp_transform, download=download, root=data_dir)
    temp_loader    = DataLoader(temp_dataset, batch_size=batch_size, shuffle=False)

    mean, std = calculate_mean_std(temp_loader)
    print(f"[BreastMNIST] Calculated Mean: {mean}, Std: {std}")

    # Step 2: Get transform pipelines
    # CNN
    transform_train_cnn, transform_val_test_cnn = get_transforms_for_cnn(mean, std)
    # RF
    transform_train_rf, transform_val_test_rf   = get_transforms_for_rf(mean, std)

    # Step 3: Load datasets and create loaders
    cnn_loaders, rf_loaders = load_breastmnist_datasets(
        batch_size=batch_size,
        download=download,
        data_dir=data_dir,
        transform_train_cnn=transform_train_cnn,
        transform_val_test_cnn=transform_val_test_cnn,
        transform_train_rf=transform_train_rf,
        transform_val_test_rf=transform_val_test_rf
    )

    # Further unpack each tuple into individual loaders
    cnn_train_loader, cnn_val_loader, cnn_test_loader = cnn_loaders
    rf_train_loader, rf_val_loader, rf_test_loader = rf_loaders

    # Step 4: Flatten features for RF
    X_train_rf, y_train_rf = flatten_features(rf_train_loader)
    X_val_rf,   y_val_rf   = flatten_features(rf_val_loader)
    X_test_rf,  y_test_rf  = flatten_features(rf_test_loader)

    # Step 5: Standardize and apply PCA
    (X_train_rf_pca, X_val_rf_pca, X_test_rf_pca), scaler, pca = preprocess_for_rf(
        X_train_rf, X_val_rf, X_test_rf, n_components
    )

    # Package RF datasets
    rf_train_data = (X_train_rf_pca, y_train_rf)
    rf_val_data   = (X_val_rf_pca,   y_val_rf)
    rf_test_data  = (X_test_rf_pca,  y_test_rf)

    return {
        "cnn_train_loader": cnn_train_loader,
        "cnn_val_loader": cnn_val_loader,
        "cnn_test_loader": cnn_test_loader,
        "rf_train_data": rf_train_data,
        "rf_val_data": rf_val_data,
        "rf_test_data": rf_test_data,
        "scaler": scaler,
        "pca": pca
    }














#--------------------------------- Task Specific Codes: Task B (Blood) ---------------------------------#

#Load BloodMNIST Datasets with Corresponding Transformations
def load_bloodmnist_datasets(
    batch_size: int,
    download: bool,
    data_dir: str,
    transform_train_cnn,
    transform_val_test_cnn,
    transform_train_rf,
    transform_val_test_rf
):
    """
    Load BloodMNIST datasets for both CNN and Random Forest with corresponding transformations.

    Args:
        batch_size (int): Number of samples per batch.
        download (bool): Whether to download the dataset if not available.
        data_dir (str): Directory where the dataset is stored or will be downloaded to.
        transform_train_cnn (transforms.Compose): Training transform for CNN.
        transform_val_test_cnn (transforms.Compose): Val/test transform for CNN.
        transform_train_rf (transforms.Compose): Training transform for RF.
        transform_val_test_rf (transforms.Compose): Val/test transform for RF.

    Returns:
        tuple: 
            - (cnn_train_loader, cnn_val_loader, cnn_test_loader)
            - (rf_train_loader, rf_val_loader, rf_test_loader)
    """
    # Load datasets for CNN
    train_dataset_cnn = BloodMNIST(split='train', transform=transform_train_cnn, download=download, root=data_dir)
    val_dataset_cnn   = BloodMNIST(split='val',   transform=transform_val_test_cnn, download=download, root=data_dir)
    test_dataset_cnn  = BloodMNIST(split='test',  transform=transform_val_test_cnn, download=download, root=data_dir)

    # Create DataLoaders for CNN
    cnn_train_loader = DataLoader(train_dataset_cnn, batch_size=batch_size, shuffle=True)
    cnn_val_loader   = DataLoader(val_dataset_cnn,   batch_size=batch_size, shuffle=False)
    cnn_test_loader  = DataLoader(test_dataset_cnn,  batch_size=batch_size, shuffle=False)

    # Load datasets for Random Forest
    train_dataset_rf = BloodMNIST(split='train', transform=transform_train_rf, download=download, root=data_dir)
    val_dataset_rf   = BloodMNIST(split='val',   transform=transform_val_test_rf, download=download, root=data_dir)
    test_dataset_rf  = BloodMNIST(split='test',  transform=transform_val_test_rf, download=download, root=data_dir)

    # Create DataLoaders for Random Forest (no shuffle needed)
    rf_train_loader = DataLoader(train_dataset_rf, batch_size=batch_size, shuffle=False)
    rf_val_loader   = DataLoader(val_dataset_rf,   batch_size=batch_size, shuffle=False)
    rf_test_loader  = DataLoader(test_dataset_rf,  batch_size=batch_size, shuffle=False)

    return (
        (cnn_train_loader, cnn_val_loader, cnn_test_loader),
        (rf_train_loader, rf_val_loader, rf_test_loader)
    )
















#Start Data Preparation for BloodMNIST
def prepare_bloodmnist_data(batch_size=32, download=True, data_dir="data_blood", n_components=50):
    """
    Prepare BloodMNIST dataset for both CNN and Random Forest.

    Steps:
    1. Compute mean/std for normalization (on training data).
    2. Define transform pipelines for CNN and RF.
    3. Load datasets and create DataLoaders for CNN and RF.
    4. Flatten features for RF.
    5. Standardize and apply PCA for RF.

    Args:
        batch_size (int, optional): Batch size. Defaults to 32.
        download (bool, optional): Download dataset if not present. Defaults to True.
        data_dir (str, optional): Directory for storing/downloading data. Defaults to "data_blood".
        n_components (int, optional): PCA components for Random Forest. Defaults to 50.

    Returns:
        tuple: 
            - cnn_train_loader, cnn_val_loader, cnn_test_loader
            - (X_train_rf_pca, y_train_rf), (X_val_rf_pca, y_val_rf), (X_test_rf_pca, y_test_rf)
            - scaler (StandardScaler), pca (PCA)
    """
    os.makedirs(data_dir, exist_ok=True)

    # Step 1: Compute mean/std from initial transform
    temp_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    temp_dataset = BloodMNIST(split='train', transform=temp_transform, download=download, root=data_dir)
    temp_loader  = DataLoader(temp_dataset, batch_size=batch_size, shuffle=False)

    mean, std = calculate_mean_std(temp_loader)
    print(f"[BloodMNIST] Calculated Mean: {mean}, Std: {std}")

    # Step 2: Define transform pipelines
    # CNN
    transform_train_cnn, transform_val_test_cnn = get_transforms_for_cnn(mean, std)
    # RF
    transform_train_rf, transform_val_test_rf   = get_transforms_for_rf(mean, std)

    # Step 3: Load datasets and create loaders
    cnn_loaders, rf_loaders = load_bloodmnist_datasets(
        batch_size=batch_size,
        download=download,
        data_dir=data_dir,
        transform_train_cnn=transform_train_cnn,
        transform_val_test_cnn=transform_val_test_cnn,
        transform_train_rf=transform_train_rf,
        transform_val_test_rf=transform_val_test_rf
    )

    # Further unpack each tuple into individual loaders
    cnn_train_loader, cnn_val_loader, cnn_test_loader = cnn_loaders
    rf_train_loader, rf_val_loader, rf_test_loader = rf_loaders

    # Step 4: Flatten features for RF
    X_train_rf, y_train_rf = flatten_features(rf_train_loader)
    X_val_rf,   y_val_rf   = flatten_features(rf_val_loader)
    X_test_rf,  y_test_rf  = flatten_features(rf_test_loader)

    # Step 5: Standardize and apply PCA
    (X_train_rf_pca, X_val_rf_pca, X_test_rf_pca), scaler, pca = preprocess_for_rf(
        X_train_rf, X_val_rf, X_test_rf, n_components
    )

    # Package RF datasets
    rf_train_data = (X_train_rf_pca, y_train_rf)
    rf_val_data   = (X_val_rf_pca,   y_val_rf)
    rf_test_data  = (X_test_rf_pca,  y_test_rf)

    return {
        "cnn_train_loader": cnn_train_loader,
        "cnn_val_loader": cnn_val_loader,
        "cnn_test_loader": cnn_test_loader,
        "rf_train_data": rf_train_data,
        "rf_val_data": rf_val_data,
        "rf_test_data": rf_test_data,
        "scaler": scaler,
        "pca": pca
    }


























if __name__ == "__main__":
    # Parameters
    BATCH_SIZE = 32
    DOWNLOAD = True
    N_COMPONENTS = 50  # Number of PCA components for Random Forest

    # ---------------------------------------
    # Preparing BreastMNIST Data (Task A)
    # ---------------------------------------
    print("--- Preparing BreastMNIST Data (Task A) ---")
    prepared_data = prepare_breastmnist_data(
        batch_size=BATCH_SIZE,
        download=DOWNLOAD,
        data_dir="data_breast",
        n_components=N_COMPONENTS
    )

    # ---------------------------------------
    # Inspect BreastMNIST CNN Training Data
    # ---------------------------------------
    print("\nBreastMNIST CNN Training Data:")
    cnn_train_loader = prepared_data["cnn_train_loader"]
    for i, (data, targets) in enumerate(cnn_train_loader):
        print(f"Batch {i+1}:")
        print(f" - Data shape: {data.shape}")        # Expected: [batch_size, channels, height, width]
        print(f" - Targets shape: {targets.shape}")  # Expected: [batch_size]
        break  # Remove this break to iterate through all batches

    # ---------------------------------------
    # Inspect BreastMNIST RF Training Data
    # ---------------------------------------
    print("\nBreastMNIST RF Training Data:")
    X_train_rf_pca, y_train_rf = prepared_data["rf_train_data"]
    print(f" - X_train_rf_pca shape: {X_train_rf_pca.shape}")  # Expected: [num_samples, n_components]
    print(f" - y_train_rf shape: {y_train_rf.shape}")            # Expected: [num_samples]

    # ---------------------------------------
    # Inspect BreastMNIST CNN Validation Data
    # ---------------------------------------
    print("\nBreastMNIST CNN Validation Data:")
    cnn_val_loader = prepared_data["cnn_val_loader"]
    for i, (data, targets) in enumerate(cnn_val_loader):
        print(f"Batch {i+1}:")
        print(f" - Data shape: {data.shape}")
        print(f" - Targets shape: {targets.shape}")
        break  # Remove this break to iterate through all batches

    # ---------------------------------------
    # Inspect BreastMNIST RF Validation Data
    # ---------------------------------------
    print("\nBreastMNIST RF Validation Data:")
    X_val_rf_pca, y_val_rf = prepared_data["rf_val_data"]
    print(f" - X_val_rf_pca shape: {X_val_rf_pca.shape}")
    print(f" - y_val_rf shape: {y_val_rf.shape}")

    # ---------------------------------------
    # Inspect BreastMNIST CNN Test Data
    # ---------------------------------------
    print("\nBreastMNIST CNN Test Data:")
    cnn_test_loader = prepared_data["cnn_test_loader"]
    for i, (data, targets) in enumerate(cnn_test_loader):
        print(f"Batch {i+1}:")
        print(f" - Data shape: {data.shape}")
        print(f" - Targets shape: {targets.shape}")
        break  # Remove this break to iterate through all batches

    # ---------------------------------------
    # Inspect BreastMNIST RF Test Data
    # ---------------------------------------
    print("\nBreastMNIST RF Test Data:")
    X_test_rf_pca, y_test_rf = prepared_data["rf_test_data"]
    print(f" - X_test_rf_pca shape: {X_test_rf_pca.shape}")
    print(f" - y_test_rf shape: {y_test_rf.shape}")

    # ---------------------------------------
    # (Optional) Proceed with Model Training and Evaluation
    # ---------------------------------------
    # At this point, you can proceed to define, train, and evaluate your CNN and Random Forest models.
    # This section is intentionally left as a placeholder for your specific implementation.

    # Example Placeholder:
    # cnn_model = YourCNNModel()
    # train_cnn(cnn_model, prepared_data["cnn_train_loader"], ...)
    # evaluate_cnn(cnn_model, prepared_data["cnn_val_loader"], ...)
    #
    # rf_model = RandomForestClassifier(n_estimators=100, ...)
    # rf_model.fit(X_train_rf_pca, y_train_rf)
    # y_pred_rf = rf_model.predict(X_val_rf_pca)
    # print(classification_report(y_val_rf, y_pred_rf))

    print("\nData preparation and inspection completed successfully.")