"""Prepare Seaborn's Titanic data for downstream machine-learning models."""

from pathlib import Path

import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


TARGET = "survived"


class DataValidationError(Exception):
    """Raised when the Titanic data does not have the required columns."""


def load_data():
    """Load the Titanic dataset included in Seaborn's example datasets."""
    data_folder = Path(__file__).resolve().parent / ".seaborn_data"
    return sns.load_dataset("titanic", data_home=data_folder)


def validate_data(data):
    """Check that the target and columns used by this pipeline are present."""
    required_columns = {TARGET, "sex", "pclass", "age", "fare", "embarked"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        names = ", ".join(sorted(missing_columns))
        raise DataValidationError(f"Required columns are missing: {names}")


def clean_text(data):
    """Trim whitespace and lowercase text values in categorical columns."""
    data = data.copy()
    for column in data.select_dtypes(include="object").columns:
        data[column] = data[column].str.strip().str.lower()
    for column in data.select_dtypes(include="category").columns:
        data[column] = data[column].astype("string").str.strip().str.lower()
    return data


def drop_useless(data):
    """Remove target leaks, duplicate columns, and mostly-empty columns."""
    data = data.copy()
    columns_to_drop = [
        "alive",       # Directly reveals whether the passenger survived.
        "deck",        # Most values are missing.
        "embark_town", # Duplicate of the embarked port.
        "class",       # Duplicate of numeric pclass.
        "who",         # Redundant with sex and age.
        "adult_male",  # Redundant with sex and age.
    ]
    return data.drop(columns=columns_to_drop, errors="ignore")


def handle_missing(data):
    """Fill numeric missing values with medians and categories with modes."""
    data = data.copy()
    for column in data.columns:
        if data[column].isna().any():
            if pd.api.types.is_numeric_dtype(data[column]):
                data[column] = data[column].fillna(data[column].median())
            else:
                modes = data[column].mode(dropna=True)
                if not modes.empty:
                    data[column] = data[column].fillna(modes.iloc[0])
                else:
                    data[column] = data[column].fillna("unknown")
    return data


def encode_categoricals(data):
    """Binary-map sex, keep passenger class ordinal, and one-hot encode ports."""
    data = data.copy()
    data["sex"] = data["sex"].map({"female": 0, "male": 1})
    # pclass is already ordinal: 1 (first) through 3 (third).
    data = pd.get_dummies(data, columns=["embarked"], dtype=int)
    return data


def split_data(data):
    """Separate features and target, then make a stratified 80/20 split."""
    features = data.drop(columns=TARGET)
    target = data[TARGET]
    return train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )


def scale_features(X_train, X_test):
    """Standardize numeric features, fitting the scaler on training data only."""
    scaler = StandardScaler()
    train_values = scaler.fit_transform(X_train)
    test_values = scaler.transform(X_test)
    return (
        pd.DataFrame(train_values, columns=X_train.columns, index=X_train.index),
        pd.DataFrame(test_values, columns=X_test.columns, index=X_test.index),
        scaler,
    )


def main():
    """Run each preparation step and report the resulting dataset sizes."""
    try:
        data = load_data()
        validate_data(data)
        data = clean_text(data)
        data = drop_useless(data)
        data = handle_missing(data)
        data = encode_categoricals(data)

        X_train, X_test, y_train, y_test = split_data(data)
        X_train, X_test, scaler = scale_features(X_train, X_test)

        print("Titanic data preparation complete.")
        print(f"Training rows: {len(X_train)}")
        print(f"Test rows: {len(X_test)}")
        print(f"Feature columns: {len(X_train.columns)}")
        return X_train, X_test, y_train, y_test, scaler
    except DataValidationError as error:
        print(f"Data validation failed: {error}")
    except Exception as error:
        print(f"Could not prepare the Titanic data: {error}")
        print("Check that seaborn, pandas, and scikit-learn are installed and the dataset is available.")


if __name__ == "__main__":
    main()

