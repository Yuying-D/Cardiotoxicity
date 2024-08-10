import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('data.csv')

def compute_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        descriptor_names = [desc[0] for desc in Descriptors._descList]
        calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
        descriptors = calculator.CalcDescriptors(mol)
        logP = Descriptors.MolLogP(mol)  # 使用MolLogP估算LogP
        mol_volume = Descriptors.MolMR(mol)  # 分子体积
        return list(descriptors) + [logP, mol_volume]
    else:
        return [None] * (len(Descriptors._descList) + 2)


properties = data['SMILES'].apply(compute_properties)
property_names = [desc[0] for desc in Descriptors._descList] + ['LogP', 'MolVolume']
properties_df = pd.DataFrame(properties.tolist(), columns=property_names)


data = pd.concat([data, properties_df], axis=1)


data['toxicity'] = data['pIC50'].apply(lambda x: 'non-toxic' if x > 5 else 'toxic')


scaler = StandardScaler()
data_standardized = data.copy()
data_standardized[property_names] = scaler.fit_transform(data[property_names])


properties_to_plot = ['MolWt', 'TPSA', 'NumHDonors', 'NumHAcceptors', 'LogP', 'MolVolume']

fig, axes = plt.subplots(2, 3, figsize=(15, 15))
axes = axes.flatten()


for i, prop in enumerate(properties_to_plot):
    sns.histplot(data_standardized, x=prop, hue='toxicity', kde=True, stat='density', common_norm=False, ax=axes[i])
    axes[i].set_title(f'Distribution of {prop}')
    axes[i].set_xlabel(prop)
    axes[i].set_ylabel('Density')
    axes[i].legend(title='Toxicity', labels=['Non-toxic', 'Toxic'])


plt.tight_layout()
plt.show()
