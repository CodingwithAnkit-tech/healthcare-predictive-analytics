import argparse
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score


# Columns for Pima dataset
NUMERIC = [
    'pregnancies', 'glucose', 'bloodpressure', 'skinthickness',
    'insulin', 'bmi', 'diabetespedigreefunction', 'age'
]

TARGET = 'outcome'


def build_pipeline(numeric_features):
    """Creates preprocessing and model pipeline"""

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, numeric_features)
    ])

    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='auc',
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5
    )

    pipe = Pipeline([
        ('pre', preprocessor),
        ('clf', model)
    ])

    return pipe


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model_out", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    X = df[NUMERIC]
    y = df[TARGET]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Build model pipeline
    model_pipeline = build_pipeline(NUMERIC)

    # Train model
    print("Training model...")
    model_pipeline.fit(X_train, y_train)

    # Evaluate
    preds = model_pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)
    print(f"Test AUROC: {auc:.4f}")

    # Save trained model
    joblib.dump(model_pipeline, args.model_out)
    print(f"Model saved to: {args.model_out}")
