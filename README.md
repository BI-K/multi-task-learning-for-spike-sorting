# Spike Sorting Multi Task

Project to evaluate whether multi-task learning is suitable for the specific data we received.


This repository focuses on **spike sorting** across multiple datasets. Each dataset contains different numbers of target classes (also referred to as "tracks"). The goal is to accurately classify the tracks using deep learning models and multi-task learning (MTL) approaches.

- We have **five different datasets**: A2, A3, A4, A12, and A21.
- We tested **classic neural network approaches** (FFN, LSTM, CNN) on **each dataset separately**.
- We then **combined all datasets** to try a **multi-task learning (MTL)** approach using FFN and CNN models.
- We observed varying performance across datasets, with datasets having more classes showing lower accuracy.

### Datasets
The data files are located in the `data` folder:
1. **A2.csv** - 2 classes/tracks
2. **A3.csv** - 2 classes/tracks
3. **A4.csv** - 3 classes/tracks
4. **A12.csv** - 6 classes/tracks
5. **A21.csv** - 2 classes/tracks

### Observations
The following resutls are from the Classic CNN:
- **A2** and **A3** (both 2 classes) typically achieve the highest accuracy (up to 98% and 94%).
- **A4** (3 classes) yields moderate accuracy (about 74%).
- **A21** (2 classes) struggles more than A2 or A3, with around 80% max accuracy.
- **A12** (6 classes) is the most challenging dataset, maxing out around 32% accuracy.

## Code Structure
- **`<model_name>.ipynb`**: Each model (FFN, CNN, LSTM) has its own Jupyter notebook. 
  - Example: `Classic-FFN-PyTorch.ipynb`, `Classic-FFN-PyTorch LSTM.ipynb`, `Classic-FFN-PyTorch CNN.ipynb`, etc.
- **`normalize_data.py`**: A helper script containing normalization functions used across the notebooks.
- Folders:
  - **`data/`**: Contains the raw CSV files (`A2.csv`, `A3.csv`, etc.).
  - **`output/`**: Any direct output files from runs, logs, or generated data can be stored here.
  - **`results/`**: Stores the final performance metrics, best configuration details, and summary CSV files.
  - **`ray_results/`**: The results obtained through ray tune Hyperparameter tuning.

## Models Tried

### Classic (Single-Task) Models
1. **Feed-Forward Network (FFN)**  
   - Trained separately on each dataset.  
   - Notebooks: `Classic-FFN-PyTorch.ipynb`.

2. **Convolutional Neural Network (CNN)**  
   - Again, trained separately for each dataset.  
   - Notebooks: `Classic-FFN-PyTorch CNN.ipynb`.

3. **Long Short-Term Memory (LSTM)**  
   - Trained separately for each dataset as well.  
   - Notebooks: `Classic-FFN-PyTorch LSTM.ipynb`.

### Multi-Task Learning (MTL) Models
1. **Feature-Based MTL with FFN**  
   - Combines all datasets in a multi-task setup, aiming to share features across tasks.  
   - Notebook: `Feature-Based-MTL.ipynb`.

2. **Feature-Based MTL with CNN**  
   - Similar approach but using a convolutional backbone for feature extraction and multi-task heads.  
   - Notebook: `Feature-Based-MTL CNN.ipynb`.

### Additional Optimization Efforts
- Specifically worked on **A12** (the underperforming dataset with 6 classes) by adding a **weighted loss** to address class imbalance or difficulty in classification.  
- Despite these changes, **A12** remains the most challenging dataset.

## Results folder Structure

A separate **results** directory is created outside the main project directory which will be shared. The structure is as follows:

```
results/
├── Classic CNN PyTorch
│   ├── Best Performance Results
│   │   └── Classic_CNN_best_performers.csv
│   ├── Full Split Results
│   │   ├── Classic_CNN_full_split_metrics_system.csv
│   │   └── Classic_CNN_one_split_metrics_system.csv
│   ├── Individual Dataset Results
│   │   ├── Classic_CNN_full_split_metrics_A12.csv
│   │   ├── Classic_CNN_full_split_metrics_A2.csv
│   │   ├── Classic_CNN_full_split_metrics_A21.csv
│   │   ├── Classic_CNN_full_split_metrics_A3.csv
│   │   └── Classic_CNN_full_split_metrics_A4.csv
│   └── Notebook
│       └── Classic-FFN-PyTorch CNN.ipynb

├── Classic FFN PyTorch
│   ├── Best Performance Results
│   │   └── Classic_FFN_best_performers.csv
│   ├── Full Split Results
│   │   ├── Classic_FFN_full_split_metrics_system.csv
│   │   └── Classic_FFN_one_split_metrics_system.csv
│   ├── Individual Dataset Results
│   │   ├── Classic_FFN_full_split_metrics_A12.csv
│   │   ├── Classic_FFN_full_split_metrics_A2.csv
│   │   ├── Classic_FFN_full_split_metrics_A21.csv
│   │   ├── Classic_FFN_full_split_metrics_A3.csv
│   │   └── Classic_FFN_full_split_metrics_A4.csv
│   └── Notebook
│       └── Classic-FFN-PyTorch.ipynb

├── Classic LSTM PyTorch
│   ├── Best Performance Results
│   │   └── Classic_LSTM_best_performers.csv
│   ├── Full Split Results
│   │   ├── Classic_LSTM_full_split_metrics_system.csv
│   │   └── Classic_LSTM_one_split_metrics_system.csv
│   ├── Individual Dataset Results
│   │   ├── Classic_LSTM_full_split_metrics_A12.csv
│   │   ├── Classic_LSTM_full_split_metrics_A2.csv
│   │   ├── Classic_LSTM_full_split_metrics_A21.csv
│   │   ├── Classic_LSTM_full_split_metrics_A3.csv
│   │   └── Classic_LSTM_full_split_metrics_A4.csv
│   └── Notebook
│       └── Classic-FFN-PyTorch LSTM.ipynb

├── Feature Based MTL CNN
│   ├── Best Performance Results
│   │   └── Feature_based_MTL_CNN_full_split_metrics_best_config.csv
│   ├── Full Split Results
│   │   ├── Feature_based_MTL_CNN_full_split_metrics_system.csv
│   │   └── Feature_based_MTL_CNN_one_split_metrics_system.csv
│   ├── Individual Dataset Results
│   │   ├── Feature_based_MTL_CNN_full_split_metrics_A12.csv
│   │   ├── Feature_based_MTL_CNN_full_split_metrics_A2.csv
│   │   ├── Feature_based_MTL_CNN_full_split_metrics_A21.csv
│   │   ├── Feature_based_MTL_CNN_full_split_metrics_A3.csv
│   │   └── Feature_based_MTL_CNN_full_split_metrics_A4.csv
│   └── Notebook
│       └── Feature-Based-MTL CNN.ipynb

└── Feature Based MTL FFN
    ├── Best Performance Results
    │   └── Feature_based_MTL_full_split_metrics_best_config.csv
    ├── Full Split Results
    │   ├── Feature_based_MTL_full_split_metrics_system.csv
    │   └── Feature_based_MTL_one_split_metrics_system.csv
    ├── Individual Dataset Results
    │   ├── Feature_based_MTL_full_split_metrics_A12.csv
    │   ├── Feature_based_MTL_full_split_metrics_A2.csv
    │   ├── Feature_based_MTL_full_split_metrics_A21.csv
    │   ├── Feature_based_MTL_full_split_metrics_A3.csv
    │   └── Feature_based_MTL_full_split_metrics_A4.csv
    └── Notebook
        └── Feature-Based-MTL.ipynb
```

### Explanation of the `results` Folder Structure
- Each **top-level folder** (e.g., **Classic CNN PyTorch**, **Classic FFN PyTorch**, etc.) corresponds to **one model approach**.
- Inside each model folder, you’ll find:
  - **Best Performance Results**: CSV files listing the best configurations or highest accuracy runs.
  - **Full Split Results**: System-level metrics on different data splits (e.g., `full_split_metrics_system.csv` for the entire dataset, `one_split_metrics_system.csv` for a single-split scenario).
  - **Individual Dataset Results**: Model performance metrics **per dataset** (A2, A3, A4, A12, A21).
  - **Notebook**: The Jupyter notebook that generated those results.

### Current Results Overview

- **A2, A3 (2 classes)**: Achieved highest accuracies (98% and 94%, respectively) with Classic CNN.
- **A4 (3 classes)**: Moderately high accuracy (~74%).
- **A21 (2 classes)**: Around 80% max accuracy, lower than expected for a 2-class dataset.
- **A12 (6 classes)**: Most challenging dataset, with a maximum accuracy of ~32% (classic CNN).
- The **weighted loss** strategy was introduced specifically for **A12** to handle the imbalance and improve performance, but gains were limited.

## Setup

1. Clone the repository:
    ```sh
    git clone https://gitlab.com/BI_Koeln/spike-sorting-multi-task.git
    cd spike-sorting-multi-task
    ```

2. Create a virtual environment (Python 3.10 required):
    ```sh
    python -m venv myenv
    ```
   - Recommended version: **Python 3.10.12**  

3. Activate the virtual environment:
    - On Windows:
        ```sh
        myenv\Scripts\activate
        ```
    - On macOS/Linux:
        ```sh
        source myenv/bin/activate
        ```

4. Install the required packages:

   - **On Windows:**
     ```sh
     pip install -r requirements_win.txt
     ```
   - **On macOS/Linux:**
     ```sh
     pip install -r requirements_mac.txt
     ```