"""
data_ucr.py  –  UCR/UEA Time Series Classification data acquisition.
Downloads and loads binary TSC datasets from the UCR archive mirror.
"""

import os
import numpy as np
import pandas as pd
import requests

DATASETS_URL = "https://raw.githubusercontent.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries/master/UCR/{name}/{name}_{split}.tsv"

HERE = os.path.dirname(__file__)
CACHE_DIR = os.path.join(HERE, "cache")

def download_ucr_dataset(name):
    """Downloads TRAIN and TEST tsv files for a given UCR dataset name."""
    os.makedirs(os.path.join(CACHE_DIR, name), exist_ok=True)
    
    for split in ["TRAIN", "TEST"]:
        dest = os.path.join(CACHE_DIR, name, f"{name}_{split}.tsv")
        if not os.path.exists(dest):
            url = DATASETS_URL.format(name=name, split=split)
            print(f"  Downloading {name} {split} from {url} ...")
            try:
                r = requests.get(url)
                r.raise_for_status()
                with open(dest, "wb") as f:
                    f.write(r.content)
            except Exception as e:
                print(f"  Error downloading {name} {split}: {e}")
                return False
    return True

def load_ucr_dataset(name):
    """Loads a UCR dataset and returns X_train, y_train, X_test, y_test."""
    # Check if extracted folder exists
    extracted_dir = os.path.join(HERE, f"{name}_extracted")
    if not os.path.exists(extracted_dir):
        if not download_ucr_dataset(name):
             raise RuntimeError(f"Could not prepare dataset {name}")
        # Note: the ZIP download logic above doesn't extract automatically in download_ucr_dataset
        # but we can assume for now we manually extracted or add extraction here.
        # For 'Yoga' specifically, we already have Yoga_extracted.
    
    # Files are usually [name]_TRAIN.txt and [name]_TEST.txt
    dfs = {}
    for split in ["TRAIN", "TEST"]:
        path = os.path.join(extracted_dir, f"{name}_{split}.txt")
        if not os.path.exists(path):
            # Try .tsv as fallback
            path = os.path.join(extracted_dir, f"{name}_{split}.tsv")
        
        print(f"  Loading {path} ...")
        # Use delim_whitespace=True for space-separated files
        df = pd.read_csv(path, sep=r'\s+', header=None)
        dfs[split] = df
        
    train = dfs["TRAIN"]
    test = dfs["TEST"]
    
    # Extract y (first column) and X (rest)
    y_train = train.iloc[:, 0].values.astype(float).astype(int)
    X_train = train.iloc[:, 1:].values.astype(float)
    
    y_test = test.iloc[:, 0].values.astype(float).astype(int)
    X_test = test.iloc[:, 1:].values.astype(float)
    
    # Map classes to {-1, 1}
    unique_classes = np.unique(y_train)
    if len(unique_classes) != 2:
        print(f"Warning: Dataset {name} has {len(unique_classes)} classes: {unique_classes}")
    
    mapping = {unique_classes[0]: -1, unique_classes[1]: 1}
    y_train = np.array([mapping[c] for c in y_train])
    y_test = np.array([mapping[c] for c in y_test])
    
    return X_train, y_train, X_test, y_test

def summarize_dataset(name, X_train, y_train, X_test, y_test):
    """Prints a summary of the dataset."""
    print(f"\nUCR Dataset: {name}")
    print(f"  Train: X={X_train.shape}, y={y_train.shape}, balance={np.mean(y_train==1):.2%}")
    print(f"  Test : X={X_test.shape}, y={y_test.shape}, balance={np.mean(y_test==1):.2%}")
    print(f"  Time series length: {X_train.shape[1]}")

if __name__ == "__main__":
    name = "Yoga"
    X_train, y_train, X_test, y_test = load_ucr_dataset(name)
    summarize_dataset(name, X_train, y_train, X_test, y_test)
