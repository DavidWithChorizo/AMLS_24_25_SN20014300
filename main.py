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
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score, make_scorer
import optuna
from optuna import Trial
from optuna.samplers import TPESampler
import csv
import datetime
from medmnist import BreastMNIST, BloodMNIST
from sklearn.feature_selection import RFE
import logging
from imblearn.over_sampling import SMOTE
import warnings
import argparse




# Suppress FutureWarnings temporarily
warnings.filterwarnings("ignore", category=FutureWarning)

#------------------------------------------------------------------- Dataset Preparation Codes -------------------------------------------------------------------#

#--------------------------------- Task Agnostic Codes ---------------------------------#

def calculate_mean_std(dataset_loader):
    """
    Calculate the mean and standard deviation for normalization based on the dataset.
    This is typically used to compute mean/std for CNN transforms if needed.
    However, we may still compute it for reference even if not strictly required by a Decision Tree.
    
    Args:
        dataset_loader (DataLoader): DataLoader for the dataset to compute statistics.

    Returns:
        tuple: (mean, std) in list form, containing mean and std per channel.
    """
    mean = 0.0
    std = 0.0
    total_samples = 0

    for inputs, _ in dataset_loader:
        batch_samples = inputs.size(0)
        mean += inputs.mean([0, 2, 3]) * batch_samples
        std += inputs.std([0, 2, 3]) * batch_samples
        total_samples += batch_samples

    mean /= total_samples
    std /= total_samples
    return mean.tolist(), std.tolist()


def flatten_features(data_loader: DataLoader):
    """
    Flatten all batches of data in a DataLoader for usage in classical ML (Decision Tree, RF, etc.).
    The images are typically [batch_size, channels, height, width], 
    so we flatten to [batch_size, channels*height*width].

    Args:
        data_loader (DataLoader): PyTorch DataLoader object.

    Returns:
        tuple: (X, y) where X is [num_samples, flattened_features], y is [num_samples].
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


def preprocess_for_rf(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray, n_components=50):
    """
    (Optional) Standardize the data and apply PCA for dimensionality reduction. 
    This can be used by any tree-based model (Decision Tree, RF, etc.) or other classical ML.

    If n_components=0, we skip PCA and only do scaling.

    Args:
        X_train (np.ndarray): Training features.
        X_val   (np.ndarray): Validation features.
        X_test  (np.ndarray): Test features.
        n_components (int): Number of PCA components (0 => skip PCA).

    Returns:
        tuple: ((X_train_processed, X_val_processed, X_test_processed), scaler, pca)
               where pca = None if n_components=0 (i.e., PCA is skipped).
    """
    # Step 1. Always do standard scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)
    X_test_scaled  = scaler.transform(X_test)

    # Step 2. (Optional) PCA
    if n_components > 0:
        pca = PCA(n_components=n_components)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_val_pca   = pca.transform(X_val_scaled)
        X_test_pca  = pca.transform(X_test_scaled)
    else:
        pca = None
        X_train_pca = X_train_scaled
        X_val_pca   = X_val_scaled
        X_test_pca  = X_test_scaled

    return (X_train_pca, X_val_pca, X_test_pca), scaler, pca


def feature_selection_rf(X_train_pca: np.ndarray, y_train: np.ndarray, 
                         X_val_pca: np.ndarray, n_features=30):
    """
    Apply Recursive Feature Elimination (RFE) using a RandomForest as the estimator.
    This is used to select the top 'n_features' based on feature importance.

    Args:
        X_train_pca (np.ndarray): Training features after scaling/PCA.
        y_train (np.ndarray): Training labels.
        X_val_pca (np.ndarray): Validation features (same shape).
        n_features (int): Number of features to select.

    Returns:
        tuple: (X_train_selected, X_val_selected, selector)
               where selector is an RFE object containing feature ranking info.
    """
    rf_estimator = RandomForestClassifier(n_estimators=100, random_state=42)
    rfe = RFE(estimator=rf_estimator, n_features_to_select=n_features, step=5)
    rfe.fit(X_train_pca, y_train)
    X_train_selected = rfe.transform(X_train_pca)
    X_val_selected   = rfe.transform(X_val_pca)
    return X_train_selected, X_val_selected, rfe


#--------------------------------- Task A: BreastMNIST + Decision Tree ---------------------------------#

def load_breastmnist_decisiontree_only(
    batch_size: int,
    download: bool,
    data_dir: str,
    transform_train,
    transform_val_test
):
    """
    For Task A: We do NOT use a CNN at all, 
    so we only load the dataset for a tree-based pipeline.

    We load (train, val, test) with the specified transforms, 
    but there's no separate path for CNN since we won't use it.

    Args:
        batch_size (int): DataLoader batch size.
        download (bool): Download if needed.
        data_dir (str): Directory path for storing data.
        transform_train (transforms.Compose): Transform for training set.
        transform_val_test (transforms.Compose): Transform for val/test set.

    Returns:
        tuple: (rf_train_loader, rf_val_loader, rf_test_loader)
    """
    from torch.utils.data import DataLoader
    
    # We only load data in one path for the Decision Tree usage
    train_dataset_rf = BreastMNIST(split='train', transform=transform_train, download=download, root=data_dir)
    val_dataset_rf   = BreastMNIST(split='val',   transform=transform_val_test, download=download, root=data_dir)
    test_dataset_rf  = BreastMNIST(split='test',  transform=transform_val_test, download=download, root=data_dir)

    rf_train_loader = DataLoader(train_dataset_rf, batch_size=batch_size, shuffle=False)
    rf_val_loader   = DataLoader(val_dataset_rf,   batch_size=batch_size, shuffle=False)
    rf_test_loader  = DataLoader(test_dataset_rf,  batch_size=batch_size, shuffle=False)

    return rf_train_loader, rf_val_loader, rf_test_loader


def prepare_breastmnist_data(
    batch_size=32, 
    download=True, 
    data_dir="data_breast",
    n_components=0,           # 0 => skip PCA, but still scale
    apply_feature_selection=False,
    n_features=20,            # Number of features to keep if RFE is used
    apply_smote=False         # New parameter for SMOTE
):
    """
    Task A: Prepare BreastMNIST dataset ONLY for a Decision Tree (or any tree-based model).
    We do NOT produce a CNN loader here because we are not using CNN for Task A.

    Steps:
    1. Compute mean/std from a minimal transform (for reference).
    2. Define transform pipelines for the 'rf'/tree approach (though we actually do scaling in code).
    3. Load datasets => Dataloaders
    4. Flatten features
    5. Scale + optional PCA => (X_train, X_val, X_test)
    6. (Optional) Apply RFE

    Args:
        batch_size (int): Batch size for the DataLoader. 
        download (bool): Whether to download if absent.
        data_dir (str): Path for storing data.
        n_components (int): PCA components, 0 => skip PCA but still do scaling.
        apply_feature_selection (bool): If True, we run RFE.
        n_features (int): Number of features to keep if RFE is used.

    Returns:
        dict: 
           {
             "X_train", "y_train",
             "X_val",   "y_val",
             "X_test",  "y_test",
             "scaler",  "pca" or None,
             "selector" if RFE used
           }
    """
    os.makedirs(data_dir, exist_ok=True)

    # Step 1. Calculate mean/std from a simple transform
    temp_transform = transforms.Compose([transforms.ToTensor()])
    from torch.utils.data import DataLoader
    temp_dataset = BreastMNIST(split='train', transform=temp_transform, download=download, root=data_dir)
    temp_loader = DataLoader(temp_dataset, batch_size=batch_size, shuffle=False)
    mean, std = calculate_mean_std(temp_loader)
    print(f"[BreastMNIST for Decision Tree] Calculated Mean: {mean}, Std: {std}")

    # Step 2. We define minimal transforms for the Decision Tree path
    # (We STILL do a ToTensor + possible normalization. But scaling is done in code anyway.)
    transform_train_rf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)  # optional, but let's keep it consistent
    ])
    transform_val_test_rf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Step 3. Load datasets => we only produce tree-based loaders
    rf_train_loader, rf_val_loader, rf_test_loader = load_breastmnist_decisiontree_only(
        batch_size=batch_size,
        download=download,
        data_dir=data_dir,
        transform_train=transform_train_rf,
        transform_val_test=transform_val_test_rf
    )

    # Step 4. Flatten
    X_train, y_train = flatten_features(rf_train_loader)
    X_val,   y_val   = flatten_features(rf_val_loader)
    X_test,  y_test  = flatten_features(rf_test_loader)

    #ravel y_train
    y_train = y_train.ravel()




    # Step 5. Scale + optional PCA
    (X_train_processed, X_val_processed, X_test_processed), scaler, pca = preprocess_for_rf(
        X_train, X_val, X_test, n_components=n_components
    )

    # Step 6. Optional RFE
    selector = None
    if apply_feature_selection:
        X_train_fs, X_val_fs, selector = feature_selection_rf(
            X_train_processed, y_train, 
            X_val_processed, 
            n_features=n_features
        )
        # overwrite
        X_train_processed = X_train_fs
        X_val_processed   = X_val_fs
        print(f"Applied RFE with top {n_features} features for Decision Tree usage.")

    # Step 7. (Optional) Apply SMOTE
    if apply_smote:
        smote = SMOTE(random_state=42)
        X_train_processed, y_train = smote.fit_resample(X_train_processed, y_train)
        print("Applied SMOTE for oversampling the minority class.")
        print("New shape after SMOTE:", X_train_processed.shape, y_train.shape)

    return {
        "X_train": X_train_processed,
        "y_train": y_train,
        "X_val":   X_val_processed,
        "y_val":   y_val,
        "X_test":  X_test_processed,
        "y_test":  y_test,
        "scaler":  scaler,
        "pca":     pca,
        "selector": selector
    }


#--------------------------------- Task B: BloodMNIST + CNN + RandomForest ---------------------------------#

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
    Load BloodMNIST datasets specifically for both CNN and Random Forest pipelines.

    Args:
        batch_size (int): Number of samples per batch.
        download (bool): Whether to download the data if not found.
        data_dir (str): Directory to store/read data.
        transform_train_cnn (transforms.Compose): Transform for CNN training set.
        transform_val_test_cnn (transforms.Compose): Transform for CNN validation/test sets.
        transform_train_rf (transforms.Compose): Transform for RF training set.
        transform_val_test_rf (transforms.Compose): Transform for RF validation/test sets.

    Returns:
        tuple: (
            (cnn_train_loader, cnn_val_loader, cnn_test_loader),
            (rf_train_loader, rf_val_loader, rf_test_loader)
        )
    """
    train_dataset_cnn = BloodMNIST(split='train', transform=transform_train_cnn, download=download, root=data_dir)
    val_dataset_cnn   = BloodMNIST(split='val', transform=transform_val_test_cnn, download=download, root=data_dir)
    test_dataset_cnn  = BloodMNIST(split='test', transform=transform_val_test_cnn, download=download, root=data_dir)

    cnn_train_loader = DataLoader(train_dataset_cnn, batch_size=batch_size, shuffle=True)
    cnn_val_loader   = DataLoader(val_dataset_cnn, batch_size=batch_size, shuffle=False)
    cnn_test_loader  = DataLoader(test_dataset_cnn, batch_size=batch_size, shuffle=False)

    train_dataset_rf = BloodMNIST(split='train', transform=transform_train_rf, download=download, root=data_dir)
    val_dataset_rf   = BloodMNIST(split='val',   transform=transform_val_test_rf, download=download, root=data_dir)
    test_dataset_rf  = BloodMNIST(split='test',  transform=transform_val_test_rf, download=download, root=data_dir)

    rf_train_loader = DataLoader(train_dataset_rf, batch_size=batch_size, shuffle=False)
    rf_val_loader   = DataLoader(val_dataset_rf, batch_size=batch_size, shuffle=False)
    rf_test_loader  = DataLoader(test_dataset_rf, batch_size=batch_size, shuffle=False)

    return (
        (cnn_train_loader, cnn_val_loader, cnn_test_loader),
        (rf_train_loader, rf_val_loader, rf_test_loader)
    )


def get_transforms_for_cnn(mean: list, std: list):
    """
    Define transformation pipeline for CNN training, including data augmentation.

    Args:
        mean (list): Mean values for normalization.
        std (list): Standard deviation values for normalization.

    Returns:
        tuple:
            - transform_train (transforms.Compose): Transformations for CNN training set.
            - transform_val_test (transforms.Compose): Transformations for CNN validation/test sets.
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


def get_transforms_for_rf(mean: list, std: list):
    """
    Define minimal transformation pipeline for Random Forest, as scaling is handled separately.

    Args:
        mean (list): Mean values for normalization.
        std (list): Standard deviation values for normalization.

    Returns:
        tuple:
            - transform_train (transforms.Compose): Transformations for RF training set.
            - transform_val_test (transforms.Compose): Transformations for RF validation/test sets.
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


def prepare_bloodmnist_data(
    batch_size=32, 
    download=True, 
    data_dir="data_blood", 
    n_components=50,
    apply_feature_selection=False,
    n_features=30,
    apply_smote=False
):
    """
    Prepare BloodMNIST data for both CNN and Random Forest pipelines.
    Includes optional PCA, RFE, and SMOTE oversampling.

    Steps:
    1. Compute dataset mean/std for normalization.
    2. Define transformation pipelines for CNN and RF.
    3. Load train/val/test data for both CNN and RF.
    4. Flatten RF data.
    5. Scale + optional PCA.
    6. Optional RFE.
    7. Optional SMOTE.

    Args:
        batch_size (int): Batch size for DataLoaders.
        download (bool): Whether to download the data if not present.
        data_dir (str): Directory path for storing data.
        n_components (int): Number of PCA components. If 0, PCA is skipped.
        apply_feature_selection (bool): Whether to apply RFE.
        n_features (int): Number of features to keep if RFE is applied.
        apply_smote (bool): If True, apply SMOTE to the RF training set.

    Returns:
        dict: {
            "cnn_train_loader", "cnn_val_loader", "cnn_test_loader",
            "rf_train_data", "rf_val_data", "rf_test_data",
            "scaler", "pca", "selector"
        }
    """
    os.makedirs(data_dir, exist_ok=True)

    # 1. Calculate mean/std from the training dataset (simple transform)
    temp_transform = transforms.Compose([transforms.ToTensor()])
    temp_loader = DataLoader(
        BloodMNIST(split='train', transform=temp_transform, download=download, root=data_dir),
        batch_size=batch_size, shuffle=False
    )
    mean, std = calculate_mean_std(temp_loader)
    print(f"[BloodMNIST] Calculated Mean: {mean}, Std: {std}")

    # 2. Define transformation pipelines
    transform_train_cnn, transform_val_test_cnn = get_transforms_for_cnn(mean, std)
    transform_train_rf, transform_val_test_rf = get_transforms_for_rf(mean, std)

    # 3. Load datasets for both CNN and RF
    cnn_loaders, rf_loaders = load_bloodmnist_datasets(
        batch_size=batch_size,
        download=download,
        data_dir=data_dir,
        transform_train_cnn=transform_train_cnn,
        transform_val_test_cnn=transform_val_test_cnn,
        transform_train_rf=transform_train_rf,
        transform_val_test_rf=transform_val_test_rf
    )

    cnn_train_loader, cnn_val_loader, cnn_test_loader = cnn_loaders
    rf_train_loader, rf_val_loader, rf_test_loader = rf_loaders

    # 4. Flatten RF data
    X_train_rf, y_train_rf = flatten_features(rf_train_loader)
    X_val_rf, y_val_rf = flatten_features(rf_val_loader)
    X_test_rf, y_test_rf = flatten_features(rf_test_loader)

    # 4.a. Ensure y is 1D
    y_train_rf = y_train_rf.ravel()
    y_val_rf = y_val_rf.ravel()
    y_test_rf = y_test_rf.ravel()

    # 5. Scale + optional PCA
    (X_train_pca, X_val_pca, X_test_pca), scaler, pca = preprocess_for_rf(
        X_train_rf, X_val_rf, X_test_rf, n_components=n_components
    )

    # 6. Optional RFE
    selector = None
    if apply_feature_selection:
        X_train_fs, X_val_fs, selector = feature_selection_rf(
            X_train_pca, y_train_rf, 
            X_val_pca, 
            n_features=n_features
        )
        X_train_pca = X_train_fs
        X_val_pca = X_val_fs
        print(f"Applied RFE with top {n_features} features for Random Forest usage.")

        # Apply RFE to Test Set if selector is available
        if selector is not None:
            X_test_pca = selector.transform(X_test_pca)
            print("Applied RFE transform to the test set.")

    # 7. Optional SMOTE on RF Training Set
    if apply_smote:
        smote = SMOTE(random_state=42)
        X_train_pca, y_train_rf = smote.fit_resample(X_train_pca, y_train_rf)
        print("Applied SMOTE to balance the RF training dataset.")

    return {
        "cnn_train_loader": cnn_train_loader,
        "cnn_val_loader": cnn_val_loader,
        "cnn_test_loader": cnn_test_loader,
        "rf_train_data": (X_train_pca, y_train_rf),
        "rf_val_data":   (X_val_pca,   y_val_rf),
        "rf_test_data":  (X_test_pca,  y_test_rf),
        "scaler":        scaler,
        "pca":           pca,
        "selector":      selector
    }

#------------------------------------------------------------------- End of Dataset Preparation Codes -------------------------------------------------------------------#












################################################################################
#                           Training and Evaluation Codes for Task A
################################################################################



#--------------------------------- Logging Setup ---------------------------------#

def setup_logging(log_file='training.log'):
    """
    Set up logging configuration.
    
    Args:
        log_file (str): Path to the log file.
    
    Returns:
        Logger object.
    """
    logging.basicConfig(
        filename=log_file,
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    return logging.getLogger()


#--------------------------------- Hyperparameter Tuning Functions ---------------------------------#
def objective(trial, X_train_dt, y_train_dt):
    # Define the hyperparameter space
    max_depth = trial.suggest_int('max_depth', 3, 50)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 30)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 30)
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
    # Example: ccp_alpha in [1e-5, 0.01], log scale
    ccp_alpha = trial.suggest_float('ccp_alpha', 1e-5, 0.01, log=True)

    # Create the Decision Tree with suggested hyperparameters
    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion,
        ccp_alpha=ccp_alpha,
        class_weight='balanced',
        random_state=42
    )

    # Evaluate using cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X_train_dt, y_train_dt, cv=skf, scoring='f1_weighted', n_jobs=-1)

    # Return the average F1-Score
    return scores.mean()




def tune_decision_tree(X_train, y_train, logger=None):
    """
    Hyperparameter tuning for Decision Tree using GridSearchCV.
    """
    # Ensure y_train is 1D
    y_train = y_train.ravel()
    
    dt = DecisionTreeClassifier(random_state=42)
    param_grid = {
        'max_depth': [None, 3, 5, 7, 10, 12, 15, 17, 20, 23, 26, 30],
        'min_samples_split': [2, 5,7, 10],
        'min_samples_leaf': [1, 2, 4, 8],
        'criterion': ['gini', 'entropy']
    }
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, 
                               cv=skf, n_jobs=-1, scoring='f1', verbose=1)
    grid_search.fit(X_train, y_train)  # y_train is now 1D
    
    if logger:
        logger.info("Decision Tree Hyperparameter Tuning Completed.")
        logger.info(f"Best Parameters: {grid_search.best_params_}")
        logger.info(f"Best F1-Score: {grid_search.best_score_}")
    else:
        print("Decision Tree Hyperparameter Tuning Completed.")
        print("Best Parameters:", grid_search.best_params_)
        print("Best F1-Score:", grid_search.best_score_)
    
    return grid_search

def tune_random_forest(X_train, y_train, logger=None):
    """
    Hyperparameter tuning for Random Forest using GridSearchCV.
    """
    # Ensure y_train is 1D
    y_train = y_train.ravel()
    
    rf = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False],
        'criterion': ['gini', 'entropy']
    }
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                               cv=skf, n_jobs=-1, scoring='f1_weighted', verbose=1)
    grid_search.fit(X_train, y_train)  # y_train is now 1D
    
    if logger:
        logger.info("Random Forest Hyperparameter Tuning Completed.")
        logger.info(f"Best Parameters: {grid_search.best_params_}")
        logger.info(f"Best F1-Score: {grid_search.best_score_}")
    else:
        print("Random Forest Hyperparameter Tuning Completed.")
        print("Best Parameters:", grid_search.best_params_)
        print("Best F1-Score:", grid_search.best_score_)
    
    return grid_search










################################################################################
#                           Training and Evaluation Codes for Task B
################################################################################


#--------------------------------- CNN Model Definition ---------------------------------#

class BloodMNIST_CNN(nn.Module):
    """
    Convolutional Neural Network tailored for BloodMNIST multi-class classification.

    Architecture:
    - 3 Convolutional layers with increasing channels
    - MaxPooling after each convolution
    - Fully connected layers with dropout
    - Output layer with softmax activation
    """

    def __init__(self, num_classes=8):
        super(BloodMNIST_CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),  # Changed to 3 channels
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 32x14x14

            nn.Conv2d(32, 64, 3, padding=1),  # Output: 64x14x14
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # Output: 64x7x7

            nn.Conv2d(64, 128, 3, padding=1),  # Output: 128x7x7
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # Output: 128x3x3
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes)
            # Note: CrossEntropyLoss in PyTorch applies Softmax internally
        )

    def forward(self, x):
        """
        Forward pass of the CNN.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 3, 28, 28].

        Returns:
            torch.Tensor: Output logits of shape [batch_size, num_classes].
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x


#--------------------------------- Logging Setup ---------------------------------#

def setup_logging(log_file='training_taskB.log'):
    """
    Configure logging to write info messages to a file.

    Args:
        log_file (str): Filename for logging output.

    Returns:
        logging.Logger: Configured logger object.
    """
    logger = logging.getLogger('TrainingLogger')
    logger.setLevel(logging.INFO)
    
    # Create handlers
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Create formatter and add to handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # Add handlers to logger
    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)
    
    return logger



#--------------------------------- Hyperparameter Tuning Functions ---------------------------------#

def objective_rf(trial, X_train_rf, y_train_rf):
    """
    Objective function for Optuna to optimize Random Forest hyperparameters.

    Args:
        trial (optuna.trial.Trial): A trial object that stores hyperparameters.
        X_train_rf (np.ndarray): Training features for Random Forest.
        y_train_rf (np.ndarray): Training labels for Random Forest.

    Returns:
        float: Mean cross-validated F1-Score of the Random Forest with the proposed hyperparameters.
    """
    # Define the hyperparameter space
    n_estimators = trial.suggest_int('n_estimators', 100, 500)
    max_depth = trial.suggest_int('max_depth', 10, 50)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
    bootstrap = trial.suggest_categorical('bootstrap', [True, False])
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])

    # Create the Random Forest with suggested hyperparameters
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        bootstrap=bootstrap,
        criterion=criterion,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    # Evaluate using cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(rf, X_train_rf, y_train_rf, cv=skf, scoring='f1_weighted', n_jobs=-1)

    # Return the average F1-Score
    return scores.mean()


def objective_cnn(trial, model, train_loader, val_loader, device):
    """
    Objective function for Optuna to optimize CNN hyperparameters.

    Args:
        trial (optuna.trial.Trial): A trial object that stores hyperparameters.
        model (nn.Module): CNN model to be trained.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        device (torch.device): Device to train the model on.

    Returns:
        float: Validation F1-Score of the CNN with the proposed hyperparameters.
    """
    # Define hyperparameter space
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD'])
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.7)

    # Modify the model's dropout rate
    model.classifier[0].p = dropout_rate
    model.classifier[3].p = dropout_rate

    # Define optimizer
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Training loop for a few epochs
    num_epochs = 5  # For faster Optuna trials; adjust as needed
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Validation
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate F1-Score
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return f1


################################################################################
#                           Training and Evaluation Functions
################################################################################

def train_cnn(model, train_loader, val_loader, device, epochs=10, learning_rate=1e-3, optimizer_type='Adam', dropout_rate=0.5, logger=None):
    """
    Train the CNN model with progress bars.

    Args:
        model (nn.Module): CNN model to be trained.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        device (torch.device): Device to train the model on.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        optimizer_type (str): Type of optimizer ('Adam' or 'SGD').
        dropout_rate (float): Dropout rate in the classifier layers.
        logger (logging.Logger): Logger object for logging info.

    Returns:
        nn.Module: Trained CNN model.
        list: Training loss history.
        list: Validation F1-Score history.
    """
    # Move model to device
    model = model.to(device)

    # Update dropout rates
    model.classifier[0].p = dropout_rate
    model.classifier[3].p = dropout_rate

    # Define optimizer
    if optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_f1_scores = []

    epoch_pbar = tqdm(range(1, epochs + 1), desc='Epoch', leave=True)
    for epoch in epoch_pbar:
        model.train()
        running_loss = 0.0

        batch_pbar = tqdm(train_loader, desc='Training', leave=False)
        for inputs, labels in batch_pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            # Update batch progress bar
            batch_pbar.set_postfix({'Batch Loss': loss.item()})

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc='Validation', leave=False)
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        f1 = f1_score(all_labels, all_preds, average='weighted')
        val_f1_scores.append(f1)

        # Update epoch progress bar
        epoch_pbar.set_postfix({'Epoch Loss': epoch_loss, 'Val F1-Score': f1})

        if logger:
            logger.info(f'Epoch {epoch}/{epochs} | Loss: {epoch_loss:.4f} | Val F1-Score: {f1:.4f}')

    return model, train_losses, val_f1_scores


def evaluate_model(model, test_loader, device, model_type='CNN'):
    """
    Evaluate the trained model on the test set.

    Args:
        model (nn.Module or sklearn estimator): Trained model.
        test_loader (DataLoader or tuple): DataLoader for CNN or tuple for RF.
        device (torch.device): Device for CNN evaluation.
        model_type (str): Type of model ('CNN' or 'RF').

    Returns:
        None
    """
    if model_type == 'CNN':
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            pbar = tqdm(test_loader, desc='Testing', leave=False)
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        print(f'\nCNN Test Results:')
        print(classification_report(all_labels, all_preds))
        # Plot Confusion Matrix
        cm = confusion_matrix(all_labels, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(8))
        disp.plot(cmap=plt.cm.Blues)
        plt.title('CNN Confusion Matrix')
        plt.show()

    elif model_type == 'RF':
        X_test_rf, y_test_rf = test_loader  # Assuming test_loader is a tuple (X_test, y_test)
        y_pred_rf = model.predict(X_test_rf)

        print(f'\nRandom Forest Test Results:')
        print(classification_report(y_test_rf, y_pred_rf))
        # Plot Confusion Matrix
        cm = confusion_matrix(y_test_rf, y_pred_rf)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(8))
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Random Forest Confusion Matrix')
        plt.show()









Task = 'B'





#--------------------------------- MAIN FUNCTION ---------------------------------#
def main():
    """
    Main function focusing on Task A: Decision Tree with BreastMNIST.
    1. Prepare data
    2. Tune Decision Tree hyperparameters
    3. Evaluate on validation & test
    4. Save the trained model (and scaler/PCA) to ./A/
    """
    if Task == 'A':
            # Setup logging
        logger = setup_logging('training_taskA.log')
        logger.info("Starting Task A: BreastMNIST + Decision Tree.")
        
        # Step 1: Prepare data for Decision Tree
        print("\n--- Preparing BreastMNIST for Decision Tree (Task A) ---")
        data_dict = prepare_breastmnist_data(
            batch_size=32,
            download=True,
            data_dir="./Datasets/BreastMNIST",
            n_components=0,              # set to 0 to skip PCA
            apply_feature_selection=False, # True if you want RFE
            n_features=20,             # number of features if RFE is applied
            apply_smote = False
        )
        
        X_train_dt = data_dict["X_train"]
        y_train_dt = data_dict["y_train"]
        X_val_dt   = data_dict["X_val"]
        y_val_dt   = data_dict["y_val"]
        X_test_dt  = data_dict["X_test"]
        y_test_dt  = data_dict["y_test"]
        
        # Confirm shapes
        print("\n[Check shapes for Decision Tree usage on BreastMNIST]")
        print(" - X_train_dt:", X_train_dt.shape)
        print(" - y_train_dt:", y_train_dt.shape)
        print(" - X_val_dt:", X_val_dt.shape)
        print(" - y_val_dt:", y_val_dt.shape)
        print(" - X_test_dt:", X_test_dt.shape)
        print(" - y_test_dt:", y_test_dt.shape)

        '''
        # Step 2: Hyperparameter Tuning with Optuna
        print("\nStarting Decision Tree Hyperparameter Tuning with Optuna.")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, X_train_dt, y_train_dt), n_trials=500, timeout=360)  # Adjust as needed

        print("Best F1-Score:", study.best_value)
        print("Best Parameters:", study.best_params)

        # Retrieve the best hyperparameters
        best_params = study.best_params

        # Create and train the best Decision Tree
        best_dt = DecisionTreeClassifier(
            max_depth=best_params['max_depth'],
            min_samples_split=best_params['min_samples_split'],
            min_samples_leaf=best_params['min_samples_leaf'],
            criterion=best_params['criterion'],
            ccp_alpha=best_params['ccp_alpha'],
            class_weight='balanced',
            random_state=42
        )
        
        best_dt.fit(X_train_dt, y_train_dt)

        # Evaluate on Validation Set
        y_val_pred_dt = best_dt.predict(X_val_dt)
        val_report = classification_report(y_val_dt, y_val_pred_dt)
        print("\nDecision Tree (Task A) Validation Results:")
        print(val_report)

        # Evaluate on Test Set
        y_test_pred_dt = best_dt.predict(X_test_dt)
        test_report = classification_report(y_test_dt, y_test_pred_dt)
        print("\nDecision Tree (Task A) Test Results:")
        print(test_report)
        '''

        # Step 2: Hyperparameter Tuning
        logger.info("Starting Decision Tree Hyperparameter Tuning.")
        dt_grid_search = tune_decision_tree(X_train_dt, y_train_dt, logger=logger)
        
        # Retrieve best Decision Tree
        best_dt = dt_grid_search.best_estimator_
        # Optionally set class_weight to handle imbalance
        best_dt.set_params(class_weight='balanced')
        
        # Retrain best Decision Tree on the entire training set
        best_dt.fit(X_train_dt, y_train_dt)
        
        # Step 3: Evaluate on Validation & Test
        print("\nDecision Tree (Task A) Validation Results:")
        y_val_pred_dt = best_dt.predict(X_val_dt)
        val_report = classification_report(y_val_dt, y_val_pred_dt)
        print(val_report)
        logger.info("Decision Tree (Task A) Validation Report:")
        logger.info(val_report)
        
        print("\nDecision Tree (Task A) Test Results:")
        y_test_pred_dt = best_dt.predict(X_test_dt)
        test_report = classification_report(y_test_dt, y_test_pred_dt)
        print(test_report)
        logger.info("Decision Tree (Task A) Test Report:")
        logger.info(test_report)
        
        # Step 4: Save the Trained Model + Preprocessing
        os.makedirs("./A", exist_ok=True)
        
        dt_model_path = "./A/best_decision_tree.joblib"
        joblib.dump(best_dt, dt_model_path)
        logger.info(f"Saved Decision Tree model to {dt_model_path}.")
        
        scaler_path = "./A/breast_scaler.joblib"
        joblib.dump(data_dict["scaler"], scaler_path)
        logger.info(f"Saved BreastMNIST scaler to {scaler_path}.")
        
        if data_dict["pca"] is not None:
            pca_path = "./A/breast_pca.joblib"
            joblib.dump(data_dict["pca"], pca_path)
            logger.info(f"Saved BreastMNIST PCA to {pca_path}.")
        
        logger.info("Completed Task A: BreastMNIST + Decision Tree.\n")
        print("\nData preparation, hyperparameter tuning, and evaluation for Task A completed successfully.")








    elif Task == 'B':
        # Parse command-line arguments
        parser = argparse.ArgumentParser(description="Train CNN and/or Random Forest on BloodMNIST.")
        parser.add_argument('--train_cnn', action='store_true', help='Flag to train the CNN model.')
        parser.add_argument('--train_rf', action='store_true', help='Flag to train the Random Forest model.')
        parser.add_argument('--epochs', type=int, default=20, help='Number of epochs for CNN training.')
        parser.add_argument('--n_trials_rf', type=int, default=50, help='Number of Optuna trials for Random Forest.')
        parser.add_argument('--n_trials_cnn', type=int, default=30, help='Number of Optuna trials for CNN hyperparameters (if implementing).')
        args = parser.parse_args()

        # Setup logging
        logger = setup_logging('training_taskB.log')
        logger.info("Starting Task B: BloodMNIST + CNN + Random Forest.")

        # 1. Prepare data
        print("\n--- Preparing BloodMNIST for CNN and Random Forest (Task B) ---")
        data_dict = prepare_bloodmnist_data(
            batch_size=32,
            download=True,
            data_dir="./Datasets/BloodMNIST",
            n_components=50,               # Number of PCA components for RF
            apply_feature_selection=True,  # Apply RFE for RF
            n_features=30,                 # Number of features to keep if RFE is applied
            apply_smote=True               # Apply SMOTE to balance RF training set
        )

        # Extract CNN DataLoaders
        cnn_train_loader = data_dict["cnn_train_loader"]
        cnn_val_loader   = data_dict["cnn_val_loader"]
        cnn_test_loader  = data_dict["cnn_test_loader"]

        # Extract RF Data
        X_train_rf, y_train_rf = data_dict["rf_train_data"]
        X_val_rf, y_val_rf     = data_dict["rf_val_data"]
        X_test_rf, y_test_rf   = data_dict["rf_test_data"]

        scaler_rf = data_dict["scaler"]
        pca_rf = data_dict["pca"]
        selector_rf = data_dict["selector"]

        # Confirm shapes for RF
        print("\n[Check shapes for Random Forest usage on BloodMNIST]")
        print(" - X_train_rf:", X_train_rf.shape)
        print(" - y_train_rf:", y_train_rf.shape)
        print(" - X_val_rf:", X_val_rf.shape)
        print(" - y_val_rf:", y_val_rf.shape)
        print(" - X_test_rf:", X_test_rf.shape)
        print(" - y_test_rf:", y_test_rf.shape)

        # 2. Define device for CNN
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"\nUsing device: {device}")

        # Initialize variables to hold models
        cnn_trained = None
        rf_trained = None

        # 3. Train CNN if selected
        if args.train_cnn:
            # Initialize CNN Model
            num_classes = 8  # Adjust based on BloodMNIST classes
            cnn_model = BloodMNIST_CNN(num_classes=num_classes)

            # Train CNN
            print("\n--- Training CNN ---")
            cnn_trained, cnn_train_losses, cnn_val_f1_scores = train_cnn(
                model=cnn_model,
                train_loader=cnn_train_loader,
                val_loader=cnn_val_loader,
                device=device,
                epochs=args.epochs,                # Number of epochs
                learning_rate=1e-3,
                optimizer_type='Adam',
                dropout_rate=0.5,
                logger=logger
            )

            # Evaluate CNN on Test Set
            evaluate_model(cnn_trained, cnn_test_loader, device, model_type='CNN')

            # Save CNN Model
            os.makedirs("./B", exist_ok=True)
            cnn_model_path = "./B/bloodmnist_cnn.pth"
            torch.save(cnn_trained.state_dict(), cnn_model_path)
            logger.info(f"Saved CNN model to {cnn_model_path}.")

        # 4. Train Random Forest if selected
        if args.train_rf:
            # Define a progress bar for Optuna trials
            print("\n--- Hyperparameter Tuning for Random Forest using Optuna ---")
            rf_study = optuna.create_study(direction='maximize', sampler=TPESampler())
            rf_pbar = tqdm(total=args.n_trials_rf, desc='RF Hyperparameter Tuning')

            def rf_callback(study, trial):
                rf_pbar.update(1)

            rf_study.optimize(lambda trial: objective_rf(trial, X_train_rf, y_train_rf), 
                            n_trials=args.n_trials_rf, 
                            timeout=None,
                            callbacks=[rf_callback])

            rf_pbar.close()

            print(f"Best Random Forest F1-Score (Optuna): {rf_study.best_value:.4f}")
            print(f"Best Parameters (Optuna): {rf_study.best_params}")

            logger.info(f"Best Random Forest F1-Score (Optuna): {rf_study.best_value:.4f}")
            logger.info(f"Best Parameters (Optuna): {rf_study.best_params}")

            # Train Best Random Forest
            best_params_rf = rf_study.best_params
            best_rf = RandomForestClassifier(
                n_estimators=best_params_rf['n_estimators'],
                max_depth=best_params_rf['max_depth'],
                min_samples_split=best_params_rf['min_samples_split'],
                min_samples_leaf=best_params_rf['min_samples_leaf'],
                bootstrap=best_params_rf['bootstrap'],
                criterion=best_params_rf['criterion'],
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            # Implement progress bar for training if desired
            print("\n--- Training Best Random Forest ---")
            rf_pbar_train = tqdm(total=1, desc='RF Training')
            best_rf.fit(X_train_rf, y_train_rf)
            rf_pbar_train.update(1)
            rf_pbar_train.close()
            logger.info("Trained Random Forest with best hyperparameters.")

            # Evaluate Random Forest on Validation Set
            print("\n--- Evaluating Random Forest on Validation Set ---")
            y_val_pred_rf = best_rf.predict(X_val_rf)
            print(classification_report(y_val_rf, y_val_pred_rf))
            logger.info("Random Forest Validation Report:")
            logger.info(classification_report(y_val_rf, y_val_pred_rf))

            # Evaluate Random Forest on Test Set
            evaluate_model(best_rf, (X_test_rf, y_test_rf), device, model_type='RF')

            # Save Random Forest Model
            os.makedirs("./B", exist_ok=True)
            rf_model_path = "./B/bloodmnist_rf.joblib"
            joblib.dump(best_rf, rf_model_path)
            logger.info(f"Saved Random Forest model to {rf_model_path}.")

        # 5. Save Preprocessing Artifacts if RF was trained
        if args.train_rf:
            os.makedirs("./B", exist_ok=True)

            # Save Scaler and PCA for RF
            scaler_path = "./B/blood_scaler.joblib"
            joblib.dump(scaler_rf, scaler_path)
            logger.info(f"Saved BloodMNIST scaler to {scaler_path}.")

            if pca_rf is not None:
                pca_path = "./B/blood_pca.joblib"
                joblib.dump(pca_rf, pca_path)
                logger.info(f"Saved BloodMNIST PCA to {pca_path}.")

            if selector_rf is not None:
                selector_path = "./B/blood_rfe.joblib"
                joblib.dump(selector_rf, selector_path)
                logger.info(f"Saved BloodMNIST RFE selector to {selector_path}.")

        logger.info("Completed Task B: BloodMNIST + CNN + Random Forest.\n")
        print("\nTask B completed successfully: Selected models trained and evaluated.")

        




















# Uncomment this if you want to auto-run the main when you do `python main.py`
if __name__ == "__main__":
     main()