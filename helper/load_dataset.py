import os
import pandas as pd

dataset_dir = os.path.join('..', os.getcwd(), 'dataset')

def load_bace_classification():
    try:
        return pd.read_csv(f'{dataset_dir}/BACE_classification.csv')
    except FileNotFoundError:
        print("File not found")

def load_bace_regression():
    try:
        return pd.read_csv(f'{dataset_dir}/BACE_regression.csv')
    except FileNotFoundError:
        print("File not found")
    
def load_esol():
    try:
        return pd.read_csv(f'{dataset_dir}/ESOL_regression.csv')
    except FileNotFoundError:
        print("File not found")
    
def load_hiv():
    try:
        return pd.read_csv(f'{dataset_dir}/HIV_classification.csv')
    except FileNotFoundError:
        print("File not found")
    
def load_lipo():
    try:
        return pd.read_csv(f'{dataset_dir}/LIPO_regression.csv')
    except FileNotFoundError:
        print("File not found")
    
def load_hdac2():
    try:
        return pd.read_csv(f'{dataset_dir}/HDAC2_classification.csv')
    except FileNotFoundError:
        print("File not found")
    

def load_fgfr1():
    try:
        return pd.read_csv(f'{dataset_dir}/FGFR1_regression.csv')
    except FileNotFoundError:
        print("File not found")