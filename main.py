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
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import optuna
from optuna import Trial
from optuna.samplers import TPESampler
import csv
import datetime
from medmnist import BreastMNIST, BloodMNIST
from sklearn.feature_selection import RFE
import logging
from sklearn.model_selection import StratifiedKFold

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
                         X_val_pca: np.ndarray, n_features=20):
    """
    Apply Recursive Feature Elimination (RFE) using a RandomForest as the estimator.
    Even if you end up training a Decision Tree, 
    we can use RF for feature ranking as it's often more stable than a single tree.

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
    n_features=20
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
    Load BloodMNIST for CNN and flatten-based approach simultaneously.
    This remains as originally coded: 
      - CNN transforms for the CNN path
      - 'RF' transforms for random forest or other classical ML
    """
    from torch.utils.data import DataLoader
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
    For Task B: transform pipeline for CNN training, includes data augmentation for train set.
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
    Minimal transforms for the RF path in BloodMNIST.
    We'll do actual scaling in code anyway.
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
    n_components=50
):
    """
    Task B: Prepare BloodMNIST for:
      - CNN usage
      - Flatten + scale + (optional) PCA for a random forest or any classical ML

    Steps:
    1. Mean/std
    2. CNN transforms
    3. Flatten
    4. Standardize + PCA
    """
    os.makedirs(data_dir, exist_ok=True)

    from torch.utils.data import DataLoader
    temp_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    temp_dataset = BloodMNIST(split='train', transform=temp_transform, download=download, root=data_dir)
    temp_loader  = DataLoader(temp_dataset, batch_size=batch_size, shuffle=False)

    mean, std = calculate_mean_std(temp_loader)
    print(f"[BloodMNIST] Calculated Mean: {mean}, Std: {std}")

    transform_train_cnn, transform_val_test_cnn = get_transforms_for_cnn(mean, std)
    transform_train_rf,  transform_val_test_rf  = get_transforms_for_rf(mean, std)

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
    rf_train_loader, rf_val_loader, rf_test_loader    = rf_loaders

    X_train_rf, y_train_rf = flatten_features(rf_train_loader)
    X_val_rf,   y_val_rf   = flatten_features(rf_val_loader)
    X_test_rf,  y_test_rf  = flatten_features(rf_test_loader)

    # Apply scaling + PCA
    (X_train_pca, X_val_pca, X_test_pca), scaler, pca = preprocess_for_rf(
        X_train_rf, X_val_rf, X_test_rf, n_components
    )

    return {
        "cnn_train_loader": cnn_train_loader,
        "cnn_val_loader": cnn_val_loader,
        "cnn_test_loader": cnn_test_loader,
        "rf_train_data": (X_train_pca, y_train_rf),
        "rf_val_data":   (X_val_pca,   y_val_rf),
        "rf_test_data":  (X_test_pca,  y_test_rf),
        "scaler":        scaler,
        "pca":           pca
    }



#------------------------------------------------------------------- End of Dataset Preparation Codes -------------------------------------------------------------------#



#--------------------------------- Training and Evaluation Codes ---------------------------------#

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
def tune_decision_tree(X_train, y_train, logger=None):
    """
    Hyperparameter tuning for Decision Tree using GridSearchCV.
    """
    # Ensure y_train is 1D
    y_train = y_train.ravel()
    
    dt = DecisionTreeClassifier(random_state=42)
    param_grid = {
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
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





















#--------------------------------- MAIN FUNCTION ---------------------------------#
def main():
    """
    Main function focusing on Task A: Decision Tree with BreastMNIST.
    1. Prepare data
    2. Tune Decision Tree hyperparameters
    3. Evaluate on validation & test
    4. Save the trained model (and scaler/PCA) to ./A/
    """
    import os
    import joblib
    from sklearn.metrics import classification_report
    
    # Setup logging
    logger = setup_logging('training_taskA.log')
    logger.info("Starting Task A: BreastMNIST + Decision Tree.")
    
    # Step 1: Prepare data for Decision Tree
    print("\n--- Preparing BreastMNIST for Decision Tree (Task A) ---")
    data_dict = prepare_breastmnist_data(
        batch_size=32,
        download=True,
        data_dir="./Datasets/BreastMNIST",
        n_components=10,              # set to 0 to skip PCA
        apply_feature_selection=False, # True if you want RFE
        n_features=10                 # number of features if RFE is applied
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

# Uncomment this if you want to auto-run the main when you do `python main.py`
if __name__ == "__main__":
     main()