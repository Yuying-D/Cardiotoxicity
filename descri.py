import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.impute import SimpleImputer


def get_rdkit_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    descriptors = [desc(mol) for name, desc in Descriptors.descList]
    return np.array(descriptors)


def load_data(filepath, use_descriptors=False):
    df = pd.read_csv(filepath)
    if use_descriptors:
        descriptor_names = [name for name, desc in Descriptors.descList]
        descriptors_list = []
        for smi in df['SMILES']:
            descriptors_list.append(get_rdkit_descriptors(smi))

        descriptors_array = np.array(descriptors_list)

        
        imputer = SimpleImputer(strategy='mean')
        descriptors_array = imputer.fit_transform(descriptors_array)

        
        descriptors_df = pd.DataFrame(descriptors_array, columns=descriptor_names)

        
        result_df = pd.concat([df['SMILES'], descriptors_df], axis=1)

        return result_df
    else:
        raise ValueError("use_descriptors = True")



input_filepath = 'processed_heart.csv'  
output_filepath = 'output_descriptors.csv' 


df_with_descriptors = load_data(input_filepath, use_descriptors=True)


df_with_descriptors.to_csv(output_filepath, index=False)
print(f"描述符已保存到 {output_filepath}")
