# AMLS_24_25_SN20014300
## IMPORTANT
** A Delayed Assessment Permit was applied and approved on this specific coursework. Therefore, the deadline is moved to the 18:00 on the 24th of January. Hence, the modifications made prior to the new deadline should be taken into account.**


## Overview

This repository contains two separate machine learning tasks (A & B) using data from **MedMNIST**:

1. **Task A**: A **Decision Tree** classifier trained on the [BreastMNIST](https://medmnist.com/) dataset.  
2. **Task B**: A **CNN** and/or **Random Forest** classifier trained on the [BloodMNIST](https://medmnist.com/) dataset.

Both tasks share common utility functions for:
- Data loading and preprocessing (flattening, normalization, optional PCA, etc.).
- Model training and evaluation pipelines.
- Hyperparameter tuning using **GridSearchCV** (with optional **Optuna** code commented out if you want to explore that).

---
## Running the code

1. **Task A: Decision Tree + BreastMNIST**
	1.	Set Task = 'A' at the top of main.py (it’s already configured, but confirm if needed).
	2.	Run the script (no special arguments for Task A):
    		```bash
        	python main.py

2. **Task B: CNN and/or Random Forest + BloodMNIST**
    1.	Set Task = 'B' at the top of main.py.
	2.	Command-line flags:
	•	python main.py --train_cnn: Trains the CNN model on BloodMNIST.
	•	python main.py --train_rf:  Trains the Random Forest model on BloodMNIST.
    3.	Examples:
	•	Train ONLY the CNN:
		```bash
		python main.py --train_cnn
	•	Train ONLY the RF:
		```bash
		 python main.py --train_rf
## Data

**BreastMNIST** and **BloodMNIST** come from the [MedMNIST](https://medmnist.com/) collection. The script will automatically download the data into the specified directories:
- `./Datasets/BreastMNIST/` for Task A
- `./Datasets/BloodMNIST/` for Task B

---

## Repository Structure
├── A/
│   ├── best_decision_tree.joblib
│   ├── breast_scaler.joblib
│   └── breast_pca.joblib
├── B/
│   ├── bloodmnist_cnn.pth
│   ├── bloodmnist_rf.joblib
│   ├── blood_scaler.joblib
│   ├── blood_pca.joblib
│   
├── Datasets/
│   ├── BreastMNIST/
│   └── BloodMNIST/
├── logs/
│   ├── training_taskA.log
│   └── training_taskB.log
├── main.py
├── README.md
├── requirements.txt
└── .gitignore


## Environment Setup

1. **Clone the repository**:
    ```bash
    git clone https:https://github.com/DavidWithChorizo/AMLS_24_25_SN20014300.git
2. **Install dependencies** (assuming you have `pip`):
    ```bash
    pip install -r requirements.txt
3.	Check GPU (if you plan to use a GPU for the CNN):
	•	PyTorch will automatically detect CUDA if available.
	•	The script also checks for mps on macOS devices.




## Running the code

1. **Task A: Decision Tree + BreastMNIST**
	1.	Set Task = 'A' at the top of main.py (it’s already configured, but confirm if needed).
	2.	Run the script (no special arguments for Task A):
        python main.py

2. **Task B: CNN and/or Random Forest + BloodMNIST**
    1.	Set Task = 'B' at the top of main.py.
	2.	Command-line flags:
	•	--train_cnn: Trains the CNN model on BloodMNIST.
	•	--train_rf:  Trains the Random Forest model on BloodMNIST.
    3.	Examples:
	•	Train ONLY the CNN:
    python main.py --train_cnn
    •	Train ONLY the RF:
    python main.py --train_rf

3. **Outputs:**
	•	CNN artifacts are saved in ./B/bloodmnist_cnn.pth.
	•	Random Forest artifacts are saved in ./B/bloodmnist_rf.joblib (plus any scaler, PCA, or RFE selectors)


## Acknowledgments
	•	MedMNIST for providing the benchmark medical image datasets.
	•	scikit-learn, PyTorch, and Optuna for the core libraries used here.
