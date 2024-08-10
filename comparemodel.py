import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_fscore_support, matthews_corrcoef


data = pd.read_csv('herg1.csv')  



def calculate_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    se = tp / (tp + fn) if (tp + fn) != 0 else 0
    sp = tn / (tn + fp) if (tn + fp) != 0 else 0
    mcc = matthews_corrcoef(y_true, y_pred)
    q = (tp + tn) / (tp + fn + tn + fp) if (tp + tn) / (tp + fn + tn + fp) != 0 else 0
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    ba = (se + sp) / 2
    auc_roc = roc_auc_score(y_true, y_pred)

    return {
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'se': se,
        'sp': sp,
        'mcc': mcc,
        'q': q,
        'P': precision,
        'F1': f1,
        'BA': ba,
        'auc_roc_score': auc_roc
    }


results = {}
for model in ['ADMETlab2.0', 'ADMETlab3.0', 'Cardpred', 'CardioDPI']:
    results[model] = calculate_metrics(data['TRUE'], data[model])


results_df = pd.DataFrame(results).T


print(results_df)
output_file = 'model_metrics1.csv' 
results_df.to_csv(output_file, index=True)
