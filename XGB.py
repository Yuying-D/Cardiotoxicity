import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, rdMolDescriptors, Descriptors
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
from sklearn.impute import SimpleImputer
import logging
from joblib import dump, load

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_fingerprint(smiles, fp_type):
    mol = Chem.MolFromSmiles(smiles)
    if fp_type == 'Morgan':
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    elif fp_type == 'MACCS':
        return MACCSkeys.GenMACCSKeys(mol)
    elif fp_type == 'AtomPairs':
        return rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol)
    elif fp_type == 'FP2':
        return Chem.RDKFingerprint(mol)
    else:
        raise ValueError("Unknown fingerprint type")

def get_rdkit_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    descriptors = [desc(mol) for name, desc in Descriptors.descList]
    return np.array(descriptors)

def load_data(filepath, fp_type=None, use_descriptors=False):
    df = pd.read_csv(filepath)
    if fp_type:
        X = np.array([get_fingerprint(smi, fp_type) for smi in df['SMILES']])
    elif use_descriptors:
        X = np.array([get_rdkit_descriptors(smi) for smi in df['SMILES']])
    else:
        raise ValueError("Either fp_type or use_descriptors must be specified")


    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    y = df['Label'].values
    return X, y

def calculate_metrics(tp, tn, fp, fn):
    se = tp / (tp + fn)
    sp = tn / (tn + fp)
    q = (tp + tn) / (tp + fn + tn + fp)
    mcc = (tp * tn - fn * fp) / np.sqrt((tp + fn) * (tp + fp) * (tn + fn) * (tn + fp))
    P = tp / (tp + fp)
    F1 = (P * se * 2) / (P + se)
    BA = (se + sp) / 2
    return {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn, 'se': se, 'sp': sp, 'mcc': mcc, 'q': q, 'P': P, 'F1': F1, 'BA': BA}

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict_proba(X_test)[:, 1]
    auc_roc_score = roc_auc_score(y_test, y_pred)
    y_pred_binary = np.round(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_binary).ravel()
    metrics = calculate_metrics(tp, tn, fp, fn)
    metrics['auc_roc_score'] = auc_roc_score

    fpr, tpr, _ = roc_curve(y_test, y_pred)
    return metrics, fpr, tpr

def save_metrics(metrics, filepath):
    df_metrics = pd.DataFrame([metrics])
    df_metrics.to_csv(filepath, index=False)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(base_dir, 'data')
    output_folder = os.path.join(base_dir, 'results')
    os.makedirs(output_folder, exist_ok=True)

    fp_types = ['Morgan', 'MACCS', 'AtomPairs', 'FP2']
    use_descriptors = True
    param_grid = {
        'max_depth': [3, 5, 7, 10, 15, 20],
        'n_estimators': [100, 200, 300]
    }
    model_name = 'XGBoost'

    metrics_all = []
    fprs_tprs = {}


    train_data_path = os.path.join(data_folder, 'train_data.csv')
    test_data_path = os.path.join(data_folder, 'test_data.csv')

    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)
    y_train = train_df['Label'].values
    y_test = test_df['Label'].values


    for fp_type in fp_types:
        X_train = np.array([get_fingerprint(smi, fp_type) for smi in train_df['SMILES']])
        X_test = np.array([get_fingerprint(smi, fp_type) for smi in test_df['SMILES']])

        model = xgb.XGBClassifier(use_label_encoder=False, random_state=42)
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        logging.info(f"Best parameters for {model_name} with {fp_type}: {grid_search.best_params_}")

        metrics, fpr, tpr = evaluate_model(model, X_test, y_test)
        metrics['model_fp_type'] = f'{model_name}::{fp_type}'
        metrics_all.append(metrics)
        fprs_tprs[f'{model_name}::{fp_type}'] = (fpr, tpr)
        save_metrics(metrics, os.path.join(output_folder, f'{model_name}_{fp_type}_metrics.csv'))


        dump(model, os.path.join(output_folder, f'{model_name}_{fp_type}_model.joblib'))

        roc_df = pd.DataFrame({'model_fp_type': f'{model_name}::{fp_type}', 'fpr': fpr, 'tpr': tpr})
        roc_df.to_csv(os.path.join(output_folder, f'{model_name}_{fp_type}_roc_data.csv'), index=False)


    if use_descriptors:
        X_train = np.array([get_rdkit_descriptors(smi) for smi in train_df['SMILES']])
        X_test = np.array([get_rdkit_descriptors(smi) for smi in test_df['SMILES']])

        model = xgb.XGBClassifier(use_label_encoder=False, random_state=42)
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        logging.info(f"Best parameters for {model_name} with RDKit Descriptors: {grid_search.best_params_}")

        metrics, fpr, tpr = evaluate_model(model, X_test, y_test)
        metrics['model_fp_type'] = f'{model_name}::RDKit_Descriptors'
        metrics_all.append(metrics)
        fprs_tprs[f'{model_name}::RDKit_Descriptors'] = (fpr, tpr)
        save_metrics(metrics, os.path.join(output_folder, f'{model_name}_RDKit_Descriptors_metrics.csv'))


        dump(model, os.path.join(output_folder, f'{model_name}_RDKit_Descriptors_model.joblib'))

        roc_df = pd.DataFrame({'model_fp_type': f'{model_name}::RDKit_Descriptors', 'fpr': fpr, 'tpr': tpr})
        roc_df.to_csv(os.path.join(output_folder, f'{model_name}_RDKit_Descriptors_roc_data.csv'), index=False)


    df_metrics_all = pd.DataFrame(metrics_all)
    df_metrics_all.to_csv(os.path.join(output_folder, 'all_xgboost_metrics.csv'), index=False)

    roc_data = []
    for model_fp, (fpr, tpr) in fprs_tprs.items():
        roc_df = pd.DataFrame({'model_fp_type': model_fp, 'fpr': fpr, 'tpr': tpr})
        roc_data.append(roc_df)
    df_roc_all = pd.concat(roc_data, ignore_index=True)
    df_roc_all.to_csv(os.path.join(output_folder, 'all_xgboost_roc_data.csv'), index=False)
