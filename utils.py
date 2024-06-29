import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

RANDOM_SEED = 42

def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def save_predictions_to_csv(y_true, y_pred, model_name, save_dir):
    df = pd.DataFrame({
        'Actual': y_true,
        'Predicted': y_pred
    })
    save_path = os.path.join(save_dir, f'{model_name}_predictions.csv')
    df.to_csv(save_path, index=False)
    print(f'Predictions saved to {save_path}')

def cross_validate_model(model, X, y, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=RANDOM_SEED)
    results = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "auc": []
    }
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro')
        rec = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X_test)
        elif hasattr(model, "decision_function"):
            prob = model.decision_function(X_test)
        else:
            prob = y_pred
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc_score = auc(fpr, tpr)
        results["accuracy"].append(acc)
        results["precision"].append(prec)
        results["recall"].append(rec)
        results["f1"].append(f1)
        results["auc"].append(auc_score)
    return {k: (np.mean(v), np.std(v)) for k, v in results.items()}
