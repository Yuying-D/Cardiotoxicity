# -*- coding: utf-8 -*-

import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

# 固定随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(2048)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
    def __init__(self, feature_size):
        super().__init__()
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=feature_size,
            nhead=64,  
            dim_feedforward=2048,
            batch_first=True
        )
        self.trans_model = nn.TransformerEncoder(
            transformer_layer,
            num_layers=4
        )
        self.classifier = nn.Linear(
            in_features=feature_size,
            out_features=1
        )
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, inputs, labels=None):
        feature = self.trans_model(inputs.unsqueeze(1))
        logits = self.classifier(feature).squeeze(1).squeeze(1)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        return logits, loss

class myDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def collate_fn(self, batch):
        features = []
        labels = []
        for example in batch:
            features.append(example["feature"])
            labels.append(example["label"])
        return torch.tensor(features).float().to(device), torch.tensor(labels).float().to(device)

def train(model, train_loader, test_loader):
    epochs = 100
    model.train()
    optimizer = Adam(model.parameters(), lr=1e-5)

    best_acc = -float('inf')
    patience = 0
    for epoch in tqdm(range(epochs)):
        print('Epoch [{}/{}]'.format(epoch + 1, epochs))

        all_loss = 0.
        for batch in tqdm(train_loader, desc="training.."):
            batch_data = [item.to(device) for item in batch]

            output, loss = model(*batch_data)  

            model.zero_grad()
            loss.backward()
            optimizer.step()
            all_loss += loss.cpu().item()

        acc = evaluate(model, test_loader)['Accuracy (ACC)']

        print("loss: {}".format(all_loss / len(train_loader)))
        print("On Epoch {}, current acc is {}, best acc is {}".format(epoch, acc, best_acc))

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'model_parameters.pth')
            patience = 0
        else:
            if patience > 20:
                break
            patience += 1

    print("Training is over, best acc is {}".format(best_acc))

def evaluate(model, data_loader):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for batch in data_loader:
            batch_data, labels = batch
            output, _ = model(batch_data)
            logits = output.squeeze()
            probs = torch.sigmoid(logits)
            preds = probs >= 0.5
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()

    
    se = recall_score(all_labels, all_preds)  
    sp = tn / (tn + fp)  
    acc = accuracy_score(all_labels, all_preds)  
    p = precision_score(all_labels, all_preds)  
    f1 = f1_score(all_labels, all_preds) 
    mcc = matthews_corrcoef(all_labels, all_preds) 
    ba = (se + sp) / 2 
    auc_value = roc_auc_score(all_labels, all_probs)  

    return {
        'Sensitivity (SE)': se,
        'Specificity (SP)': sp,
        'Accuracy (ACC)': acc,
        'Precision (P)': p,
        'F1 Score (F1)': f1,
        'Matthews Correlation Coefficient (MCC)': mcc,
        'Balanced Accuracy (BA)': ba,
        'Area Under Curve (AUC)': auc_value
    }

if __name__ == "__main__":

    file_name = "./fingerprints/morgan_fingerprints.csv"
    df = pd.read_csv(file_name)
    num_columns = df.shape[1] - 1

    data = []
    for idx, line in df.iterrows():
        data.append(
            {
                "feature": line.tolist()[:-1],
                "label": line["Label"]
            }
        )
    random.shuffle(data)

  
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    test_data = data[train_size:]

    train_dataset = myDataset(train_data)
    test_dataset = myDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=train_dataset.collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=test_dataset.collate_fn)

    model = Model(num_columns).to(device)
    train(model, train_loader, test_loader)


    model.load_state_dict(torch.load('model_parameters.pth'))

    test_metrics = evaluate(model, test_loader)
    print("Test set evaluation:")
    for key, value in test_metrics.items():
        print(f"{key}: {value}")


    metrics_df = pd.DataFrame([test_metrics], index=["Test"])
    metrics_df.to_csv('evaluation_metrics_test.csv', index=True)
超参数调优