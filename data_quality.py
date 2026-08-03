import numpy as np
import pandas as pd


DATA_PATH = "data/data.csv"
TARGET_COLUMN = "Target"
TARGET_MAP = {"Dropout": 0, "Enrolled": 1, "Graduate": 2}
CLASS_NAMES = list(TARGET_MAP.keys())

# These columns store *category codes*, not quantities (e.g. Course 171 is a
# different major than Course 17, not "10x" of it). Feeding the raw code to a
# linear model implies a false ordering/magnitude, so we one-hot encode them.
#
# Only the LOW-cardinality codes are encoded. 5-fold CV showed these two lift
# Logistic Regression from ~73.9% -> ~75.2%, while also one-hot encoding the
# high-cardinality codes (parental occupation ~46 levels, qualifications ~30,
# nationality ~21) explodes into hundreds of sparse columns that OVERFIT and
# drop accuracy to ~70%. Application mode tested neutral, so it's left out.
# Already-binary flags (Gender, Debtor, ...) and ordinal codes are left as-is.
NOMINAL_CATEGORICAL_COLUMNS = [
    "Marital status",
    "Course",
]


def load_raw_data(path=DATA_PATH):
    return pd.read_csv(path, sep=";")


def clean_data(df):
    cleaned = df.copy()
    actions = []

    original_columns = list(cleaned.columns)
    cleaned.columns = [str(column).strip() for column in cleaned.columns]
    renamed_columns = {
        old: new
        for old, new in zip(original_columns, cleaned.columns)
        if old != new
    }
    if renamed_columns:
        actions.append(
            f"Cleaned whitespace/control characters from {len(renamed_columns)} column name(s)."
        )

    if TARGET_COLUMN not in cleaned.columns:
        raise ValueError(f"Missing required target column: {TARGET_COLUMN}")

    if cleaned[TARGET_COLUMN].dtype == "object":
        cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].str.strip()
        actions.append("Trimmed whitespace from target labels.")

    duplicate_count = int(cleaned.duplicated().sum())
    if duplicate_count:
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)
        actions.append(f"Removed {duplicate_count} duplicate row(s).")

    invalid_targets = sorted(set(cleaned[TARGET_COLUMN].dropna()) - set(TARGET_MAP))
    if invalid_targets:
        raise ValueError(f"Unexpected target label(s): {invalid_targets}")

    feature_columns = [column for column in cleaned.columns if column != TARGET_COLUMN]
    non_numeric_features = cleaned[feature_columns].select_dtypes(exclude="number").columns.tolist()
    if non_numeric_features:
        cleaned[non_numeric_features] = cleaned[non_numeric_features].apply(
            pd.to_numeric, errors="coerce"
        )
        actions.append(f"Converted {len(non_numeric_features)} feature column(s) to numeric.")

    missing_by_feature = cleaned[feature_columns].isna().sum()
    missing_feature_columns = missing_by_feature[missing_by_feature > 0]
    if not missing_feature_columns.empty:
        medians = cleaned[feature_columns].median(numeric_only=True)
        cleaned[feature_columns] = cleaned[feature_columns].fillna(medians)
        actions.append(
            f"Filled missing numeric feature values in {len(missing_feature_columns)} column(s) with medians."
        )

    cleaned, engineered = engineer_curricular_features(cleaned)
    if engineered:
        actions.append(
            f"Engineered {len(engineered)} curricular-unit feature(s): {', '.join(engineered)}."
        )

    cleaned, encoded_cols, dummy_count = one_hot_encode_categoricals(cleaned)
    if encoded_cols:
        actions.append(
            f"One-hot encoded {len(encoded_cols)} categorical code column(s) "
            f"into {dummy_count} indicator column(s)."
        )

    if not actions:
        actions.append("No cleaning changes were required.")

    return cleaned, actions


def one_hot_encode_categoricals(df, columns=NOMINAL_CATEGORICAL_COLUMNS):
    """Replace nominal category-code columns with 0/1 indicator columns.

    A linear model reads 'Course = 171' as a large quantity; one-hot encoding
    turns it into separate yes/no columns (Course_171, Course_17, ...) so the
    codes are treated as unordered categories. Encoding the full dataset here
    (before any train/test split) guarantees every consumer -- model.py,
    evaluate.py, app.py -- sees an identical set of columns.

    Returns (df, encoded_source_columns, total_indicator_columns).
    """
    present = [c for c in columns if c in df.columns]
    if not present:
        return df, [], 0

    # Codes are integers; cast so dummy names read 'Course_171', not 'Course_171.0'.
    df[present] = df[present].astype(int)

    before = df.shape[1]
    df = pd.get_dummies(df, columns=present, dtype=int)
    dummy_count = df.shape[1] - (before - len(present))

    return df, present, dummy_count


def engineer_curricular_features(df):
    """Derive relationship features from the per-semester curricular-unit columns.

    The raw columns store enrolled/approved/grade as separate counts, but the
    predictive signal lives in the *relationship* between them (e.g. passing 6
    of 6 vs 6 of 12). Linear models can't recover a ratio from separate columns,
    so we compute it explicitly. Returns (df, list_of_new_column_names).
    """
    engineered = []

    ratio_specs = [
        ("Curricular units 1st sem (approved)", "Curricular units 1st sem (enrolled)", "1st sem approval ratio"),
        ("Curricular units 2nd sem (approved)", "Curricular units 2nd sem (enrolled)", "2nd sem approval ratio"),
    ]
    for approved_col, enrolled_col, new_col in ratio_specs:
        if approved_col in df.columns and enrolled_col in df.columns:
            # Guard divide-by-zero (student enrolled in nothing) -> ratio 0.
            df[new_col] = (
                df[approved_col] / df[enrolled_col].replace(0, np.nan)
            ).fillna(0)
            engineered.append(new_col)

    grade_1st, grade_2nd = (
        "Curricular units 1st sem (grade)",
        "Curricular units 2nd sem (grade)",
    )
    if grade_1st in df.columns and grade_2nd in df.columns:
        # Momentum between semesters: a sliding grade signals rising risk.
        df["grade delta"] = df[grade_2nd] - df[grade_1st]
        engineered.append("grade delta")

    return df, engineered


def class_imbalance_report(df):
    counts = df[TARGET_COLUMN].value_counts().reindex(CLASS_NAMES, fill_value=0)
    percentages = (counts / counts.sum() * 100).round(2)
    minority_count = int(counts.min())
    majority_count = int(counts.max())
    majority_minority_ratio = (
        round(majority_count / minority_count, 2) if minority_count else float("inf")
    )
    minority_percentage = float(percentages.min())
    is_imbalanced = minority_percentage < 20 or majority_minority_ratio >= 2

    return {
        "counts": counts,
        "percentages": percentages,
        "minority_class": counts.idxmin(),
        "majority_class": counts.idxmax(),
        "minority_percentage": minority_percentage,
        "majority_minority_ratio": majority_minority_ratio,
        "is_imbalanced": is_imbalanced,
    }


def audit_data(df):
    feature_columns = [column for column in df.columns if column != TARGET_COLUMN]
    missing_by_column = df.isna().sum()
    missing_by_column = missing_by_column[missing_by_column > 0].sort_values(ascending=False)

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_cells": int(df.isna().sum().sum()),
        "missing_by_column": missing_by_column,
        "non_numeric_features": df[feature_columns].select_dtypes(exclude="number").columns.tolist(),
        "constant_features": [
            column for column in feature_columns if df[column].nunique(dropna=False) <= 1
        ],
        "dirty_column_names": [
            column for column in df.columns if str(column).strip() != str(column)
        ],
        "unexpected_targets": (
            sorted(set(df[TARGET_COLUMN].dropna()) - set(TARGET_MAP))
            if TARGET_COLUMN in df.columns
            else []
        ),
        "class_balance": (
            class_imbalance_report(df) if TARGET_COLUMN in df.columns else None
        ),
    }


def load_clean_data(path=DATA_PATH, map_target=False):
    raw = load_raw_data(path)
    raw_audit = audit_data(raw)
    cleaned, cleaning_actions = clean_data(raw)
    cleaned_audit = audit_data(cleaned)

    if map_target:
        cleaned = cleaned.copy()
        cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].map(TARGET_MAP)

    return cleaned, {
        "raw_audit": raw_audit,
        "cleaned_audit": cleaned_audit,
        "cleaning_actions": cleaning_actions,
    }


def print_audit_report(audit_details):
    raw = audit_details["raw_audit"]
    cleaned = audit_details["cleaned_audit"]
    balance = cleaned["class_balance"]

    print("\n########## DATA AUDIT, CLEANING, AND CLASS BALANCE ##########")
    print(f"Raw dataset: {raw['rows']} rows x {raw['columns']} columns")
    print(f"Cleaned dataset: {cleaned['rows']} rows x {cleaned['columns']} columns")
    print(f"Duplicate rows before cleaning: {raw['duplicate_rows']}")
    print(f"Missing cells after cleaning: {cleaned['missing_cells']}")

    print("\nCleaning actions:")
    for action in audit_details["cleaning_actions"]:
        print(f"- {action}")

    print("\nRemaining audit flags after cleaning:")
    print(f"- Dirty column names: {cleaned['dirty_column_names'] or 'None'}")
    print(f"- Non-numeric feature columns: {cleaned['non_numeric_features'] or 'None'}")
    print(f"- Constant feature columns: {cleaned['constant_features'] or 'None'}")
    print(f"- Unexpected target labels: {cleaned['unexpected_targets'] or 'None'}")

    print("\nClass balance:")
    for class_name in CLASS_NAMES:
        print(
            f"- {class_name}: {balance['counts'][class_name]} "
            f"({balance['percentages'][class_name]:.2f}%)"
        )
    print(f"Majority class: {balance['majority_class']}")
    print(f"Minority class: {balance['minority_class']}")
    print(f"Majority/minority ratio: {balance['majority_minority_ratio']}:1")
    if balance["is_imbalanced"]:
        print("Imbalance warning: minority class is under 20% or ratio is at least 2:1.")
    else:
        print("No class imbalance warning triggered.")
