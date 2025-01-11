# This is the main file for the project. It will be used to run the project. 
# There will be two models being trained for the first part and of the project, and two for the latter part.

#---------------------------------------------------- Task 1, Model 1 ----------------------------------------------------#

# Prior to running the code, the following libraries need to be installed:

# -------------------------------------------------------------------
# 1. Standard Libraries
# -------------------------------------------------------------------
import os
import sys
import warnings
from pathlib import Path

# -------------------------------------------------------------------
# 2. Numerical and Data Handling
# -------------------------------------------------------------------
import numpy as np
import pandas as pd
import scipy

# -------------------------------------------------------------------
# 3. Data Visualization
# -------------------------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns


# -------------------------------------------------------------------
# 4. Image Processing
# -------------------------------------------------------------------
from PIL import Image
import cv2
import skimage
from skimage import io, transform

# -------------------------------------------------------------------
# 5. Machine Learning Libraries
# -------------------------------------------------------------------
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE

# -------------------------------------------------------------------
# 6. Deep Learning with PyTorch
# -------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms, datasets, models

# -------------------------------------------------------------------
# 7. Advanced Visualization and Utilities
# -------------------------------------------------------------------
from torchsummary import summary
from torchviz import make_dot
from tqdm.notebook import tqdm  # For progress bars in Jupyter
import matplotlib.gridspec as gridspec




# We then need to import the datasets from the medmnist website. 
from medmnist import BreastMNIST, BloodMNIST
