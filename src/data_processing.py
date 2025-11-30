import argparse
import pandas as pd

def load_and_clean(path):
    df = pd.read_csv(path)

    # Column names cleanup
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Replace zero values with NaN (for medical datasets)
    zero_as_nan = ['glucose', 'bloodpressure', 'skinthickness', 'insulin', 'bmi']
    for col in zero_as_nan:
        if col in df.columns:
            df[col] = df[col].replace(0, pd.NA)

    # Fill missing values with median
    df = df.fillna(df.median(numeric_only=True))

    return df

def save_processed(df, outpath):
    df.to_csv(outpath, index=False)
    print(f"Processed file saved to: {outpath}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    df = load_and_clean(args.input)
    save_processed(df, args.out)
