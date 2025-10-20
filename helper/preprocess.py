import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmilesFromSmiles
from collections import defaultdict
# from helper.graphfeat import StructureEncoder
import torch_geometric.data as data

def _generate_scaffold(smi:str, include_chirality:bool=True) -> str:
    return MurckoScaffoldSmilesFromSmiles(smi, includeChirality=include_chirality)
    
def split_train_test(dataset:pd.DataFrame, smiles_col:int='SMILES', type:str='random', test_size:int=0.1, random_state:int=42, shuffle:bool=True) -> pd.DataFrame:
    if type == 'random':
        train, test = train_test_split(dataset, test_size=test_size, random_state=random_state, shuffle=shuffle)
        
    elif type == 'scaffold':
        dataset['scaffold'] = dataset[smiles_col].apply(_generate_scaffold)
        scaffold_sets = defaultdict(list)
        for idx, scaffold in enumerate(dataset['scaffold']):
            scaffold_sets[scaffold].append(idx)
        scaffold_sets = {key: sorted(value) for key, value in scaffold_sets.items()}
        scaffold_lists = [
            scaffold_set for (scaffold, scaffold_set) in sorted(scaffold_sets.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
        ]
        len_dataset = len(dataset)
        train_cutoff = (1-test_size) * len_dataset
        train_idx = []
        test_idx = []
        for scaffold_set in scaffold_lists:
            if len(train_idx) + len(scaffold_set) > train_cutoff:
                test_idx += scaffold_set
            else:
                train_idx += scaffold_set
        train = dataset.iloc[train_idx]
        test = dataset.iloc[test_idx]
    
    else:
        raise ValueError('No split type provided, supported: random | scaffold')

    return train, test

def split_train_valid_test(dataset:pd.DataFrame, smiles_col:int='SMILES', type:str='random', test_size:int=0.1, valid_size:int=0.1, random_state:int=42, shuffle:bool=True) -> pd.DataFrame:
    if type == 'random':
        valid_size = valid_size * 1.25
        temp, test = train_test_split(dataset, test_size=test_size, random_state=random_state, shuffle=shuffle)
        train, valid = train_test_split(temp, test_size=valid_size, random_state=random_state, shuffle=shuffle)
        
    elif type == 'scaffold':
        dataset['scaffold'] = dataset[smiles_col].apply(_generate_scaffold)
        scaffold_sets = defaultdict(list)
        for idx, scaffold in enumerate(dataset['scaffold']):
            scaffold_sets[scaffold].append(idx)
        scaffold_sets = {key: sorted(value) for key, value in scaffold_sets.items()}
        scaffold_lists = [
            scaffold_set for (scaffold, scaffold_set) in sorted(scaffold_sets.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
        ]
        len_dataset = len(dataset)
        train_cutoff = (1-valid_size-test_size) * len_dataset
        valid_cutoff = (1-test_size) * len_dataset
        train_idx = []
        valid_idx = []
        test_idx = []
        for scaffold_set in scaffold_lists:
            if len(train_idx) + len(scaffold_set) > train_cutoff:
                if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                    test_idx += scaffold_set
                else:
                    valid_idx += scaffold_set
            else:
                train_idx += scaffold_set
        train = dataset.iloc[train_idx]
        valid = dataset.iloc[valid_idx]
        test = dataset.iloc[test_idx]

    else:
        raise ValueError('No split type provided, supported: random | scaffold')
    
    return train, valid, test

def generate_graph_dataset(dataset, smi_col, target_col, encoder):
    graph_generation = encoder
    graph_list = []
    smis = dataset[smi_col]
    labels = dataset[target_col]
    for smi, label in zip(smis, labels):
        graph = graph_generation.encoding_structure(smi, label)
        graph_list.append(graph)
    return data.Batch.from_data_list(graph_list)