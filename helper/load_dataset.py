import os
import pandas as pd

dataset_dir = os.path.join('..', os.getcwd(), 'dataset')

def load_bace_classification():
    try:
        return pd.read_csv(f'{dataset_dir}/BACE_classification.csv')
    except:
        raise FileNotFoundError()

def load_bace_regression():
    try:
        return pd.read_csv(f'{dataset_dir}/BACE_regression.csv')
    except:
        raise FileNotFoundError()
    
def load_esol():
    try:
        return pd.read_csv(f'{dataset_dir}/ESOL_regression.csv')
    except:
        raise FileNotFoundError()
    
def load_hiv():
    try:
        return pd.read_csv(f'{dataset_dir}/HIV_classification.csv')
    except:
        raise FileNotFoundError()
    
def load_lipo():
    try:
        return pd.read_csv(f'{dataset_dir}/LIPO_regression.csv')
    except:
        raise FileNotFoundError()