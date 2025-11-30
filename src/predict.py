import argparse
import pandas as pd
import joblib


def load_model(model_path):
    """Load trained ML model"""
    return joblib.load(model_path)


def make_prediction(model, input_data):
    """Generate prediction"""
    probabilities = model.predict_proba(input_data)[:, 1]
    return probabilities


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--input", required=True, help="CSV file with input data")
    args = parser.parse_args()

    # Load model
    model = load_model(args.model)

    # Load input CSV
    df = pd.read_csv(args.input)

    # Predict
    preds = make_prediction(model, df)

    print("Predicted Probabilities (Disease Risk):")
    for i, p in enumerate(preds):
        print(f"Sample {i+1}: {p:.4f}")
