import argparse
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def _save_series(series: pd.Series, path: Path) -> None:
    """Persist a pandas Series to CSV while preserving its name as the header."""
    header = series.name or "target"
    series.to_frame(name=header).to_csv(path, index=False)


def preprocess_data(data: pd.DataFrame, save_path: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Preprocess the provided dataset and persist the resulting artifacts.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset containing a `Test Results` target column.
    save_path : str | Path
        Directory where the processed datasets and preprocessor will be stored.
    """

    output_dir = Path(save_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    if "Test Results" not in data.columns:
        raise KeyError("Input data must contain a 'Test Results' column.")

    data = data.drop(['Billing Amount', 'Room Number', 'Name', 'Date of Admission', 'Doctor', 'Hospital', 'Discharge Date'], axis=1)
    
    X = data.drop(columns=["Test Results"])
    y = data["Test Results"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    transformer = make_column_transformer(
        (
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            make_column_selector(dtype_exclude=np.number),
        ),
        remainder="passthrough",
    ).set_output(transform="pandas")

    X_train = transformer.fit_transform(X_train, y_train)
    X_test = transformer.transform(X_test)

    joblib.dump(transformer, output_dir / "preprocessor.joblib")

    X_train.to_csv(output_dir / "X_train.csv", index=False)
    X_test.to_csv(output_dir / "X_test.csv", index=False)
    _save_series(y_train, output_dir / "y_train.csv")
    _save_series(y_test, output_dir / "y_test.csv")

    return X_train, X_test, y_train, y_test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess dataset and persist artifacts.")

    script_dir = Path(__file__).resolve().parent
    default_input = script_dir.parent / "healthcare_dataset.csv"
    default_output_dir = script_dir / "preprocessed"

    parser.add_argument(
        "--input",
        default=default_input,
        type=Path,
        help="Path to the input dataset in CSV format.",
    )
    parser.add_argument(
        "--output-dir",
        default=default_output_dir,
        type=Path,
        help="Directory where preprocessed datasets and artifacts will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.input)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(data_path)
    preprocess_data(data, output_dir)


if __name__ == "__main__":
    main()

