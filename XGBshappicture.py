import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from sklearn.model_selection import train_test_split
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import os
from tqdm import tqdm



def fragment_search(smiles, bit_idx, image_dir, molecules_index):
    for index, mol_fingerprint in enumerate(np.array(X)):
        if mol_fingerprint[bit_idx] == 1 and index not in molecules_index:
            mol = Chem.MolFromSmiles(smiles.iloc[index])
            bi = {}
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024, bitInfo=bi)
            try:
                img_bytes = Draw.DrawMorganBit(mol, bit_idx, bi, useSVG=False)
                with open(os.path.join(image_dir, f'molecule_{index}_Morgan_{bit_idx}_fragment.png'), 'wb') as img_file:
                    img_file.write(img_bytes)
                Draw.MolToFile(mol, os.path.join(image_dir, f'molecule_{index}_Morgan_{bit_idx}.png'))
                molecules_index.append(index)
                break
            except Exception as e:
                print(f"Error drawing fragment for molecule {index}: {e}")
                continue
    return None



file = 'xgboost_test'
if not os.path.exists('./' + file):
    os.makedirs('./' + file)
    os.makedirs('./' + file + '/images')
image_dir = ('./' + file + '/images')


file_path = '/Users/chz/Desktop/GPT/heart/heart碎片材料/ML/ronghe/data.csv'  
df = pd.read_csv(file_path)


print(df.head())
print(df.columns)


X = np.array([AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi), 2, nBits=1024)
              for smi in tqdm(df['SMILES'], desc='Generating Morgan fingerprints')])


feature_names = ['Morgan_' + str(s) for s in range(0, 1024)]


y = df['Label'].values  


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
X_train = pd.DataFrame(X_train, columns=feature_names)


xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)


explainer = shap.Explainer(xgb_model, X_train)
shap_values = explainer(X_test)

# 绘制SHAP summary图
shap.summary_plot(shap_values, X_test)
plt.title('SHAP Summary')
plt.savefig(f'./{file}/shap_summary.png', bbox_inches='tight')
plt.clf()

# 绘制SHAP dependence图
for feature_idx in range(20):
    shap.dependence_plot(feature_idx, shap_values.values, X_test, show=False)
    plt.savefig(f'./{file}/shap_dependence_feature_{feature_idx}.png', bbox_inches='tight')
    plt.clf()

# 计算top20特征 得到list
top_20_features = list(
    pd.DataFrame(abs(shap_values.values), columns=feature_names).mean().sort_values(ascending=False).head(20).index)

important_fragment = []

for feature_i in top_20_features:
    important_fragment.append(int(feature_i[7:]))

molecules_index = []
for i in important_fragment:
    fragment_search(df['SMILES'], i, image_dir, molecules_index)

print("All tasks completed successfully.")
