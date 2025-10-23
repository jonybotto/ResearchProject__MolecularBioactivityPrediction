import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import MACCSkeys
from rdkit.Chem import Descriptors
from rdkit.Chem import rdReducedGraphs
from typing import List

def smi_ecfp(smi:str, radius:int=2, n_bits:int=1024) -> List[int]:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return np.zeros((n_bits,))
    fpGen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = fpGen.GetFingerprint(mol)
    arr = np.zeros((1,))
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def smi_maccs(smi:str) -> List[int]:
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        maccs = list(MACCSkeys.GenMACCSKeys(mol))
        return maccs
    else:
        return None
    
def smi_rdkitDesc(smi:str) -> List[int]:
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        mol_desc = Descriptors.CalcMolDescriptors(mol)
        desc_list = []
        for desc in mol_desc.items():
            desc_list.append(desc[1])
        return desc_list
    else:
        return None

def smi_erg(smi:str) -> List[int]:
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        return rdReducedGraphs.GetErGFingerprint(mol)
    else:
        return None
        