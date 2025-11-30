import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, auc
)

def evaluate_model(model, X, y):
    # Predict probabilities and classes
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    print("\n===== MODEL EVALUATION =====")
    print("ROC-AUC Score:", roc_auc_score(y, y_prob))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y, y_pred))

    print("\nClassification Report:")
    print(classification_report(y, y_pred))

    precision, recall, _ = precision_recall_curve(y, y_prob)
    print("PR-AUC:", auc(recall, precision))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="processed CSV")
    parser.add_argument("--model", required=True, help="model file path")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    X = df.drop(columns=['outcome'])
    y = df['outcome']

    model = joblib.load(args.model)

    evaluate_model(model, X, y)
