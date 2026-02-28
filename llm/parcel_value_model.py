from __future__ import annotations

"""
Simple sklearn-based land-value model over the parcel dataset.

This is intentionally lightweight: it wraps training + prediction in one place
so the API layer can just call `predict_land_value_for_parcel_id`.
"""

import json
import os
from functools import lru_cache
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@lru_cache(maxsize=1)
def _load_parcel_dataframe() -> pd.DataFrame:
    """Load the default JSON parcel dataset into a DataFrame."""
    # Reuse the same resolution that RAG uses.
    from llm.rag import DEFAULT_DATASET_FILENAME

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    app_path = os.path.join(project_root, "app", DEFAULT_DATASET_FILENAME)
    root_path = os.path.join(project_root, DEFAULT_DATASET_FILENAME)
    dataset_path = app_path if os.path.exists(app_path) else root_path
    if not os.path.exists(dataset_path):
        raise RuntimeError(f"Parcel dataset not found at {dataset_path}")

    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise RuntimeError("Parcel dataset JSON must be an array of records.")
    return pd.DataFrame(data)


@lru_cache(maxsize=1)
def _train_model() -> Pipeline:
    """
    Train a very simple RandomForestRegressor to predict Land Value
    from a handful of basic features. This is not meant to be a
    production-grade model, just a demo of sklearn integration.
    """
    df = _load_parcel_dataframe()
    if "Land Value" not in df.columns:
        raise RuntimeError("Parcel dataset does not contain 'Land Value' column.")

    # Target
    y_raw = df["Land Value"]

    def _to_float(series: pd.Series) -> pd.Series:
        return (
            series.astype(str)
            .str.replace(",", "", regex=False)
            .str.strip()
            .replace("", np.nan)
            .astype(float)
        )

    y = _to_float(y_raw)

    # Basic numeric + categorical features.
    feature_cols_num = []
    feature_cols_cat = []
    if "Acres" in df.columns:
        feature_cols_num.append("Acres")
    if "Last Sale Date" in df.columns:
        # We won't use the raw date, but we can count non-missing as a signal.
        df["has_last_sale"] = df["Last Sale Date"].notna().astype(int)
        feature_cols_num.append("has_last_sale")
    if "Land Use / Use Code" in df.columns:
        feature_cols_cat.append("Land Use / Use Code")

    if not feature_cols_num and not feature_cols_cat:
        raise RuntimeError("Not enough usable columns to train a value model.")

    X = df[feature_cols_num + feature_cols_cat].copy()

    numeric_transformer = "passthrough"
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, feature_cols_num),
            ("cat", categorical_transformer, feature_cols_cat),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=60,
        random_state=42,
        n_jobs=-1,
    )

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    # Drop rows with missing target.
    mask = y.notna()
    X_train = X[mask]
    y_train = y[mask]
    if len(X_train) < 50:
        raise RuntimeError("Not enough rows with Land Value to train a model.")

    pipe.fit(X_train, y_train)
    return pipe


def predict_land_value_for_parcel_id(parcel_id: str) -> Optional[float]:
    """
    Predict land value for a specific parcel id using the trained sklearn model.

    Returns a numeric prediction (float) or None if the parcel cannot be found
    or if the model cannot be trained.
    """
    if not parcel_id:
        return None
    try:
        df = _load_parcel_dataframe()
        pipe = _train_model()
    except Exception:
        return None

    mask = (df.get("Parcel ID") == parcel_id) | (df.get("Parcel ID ") == parcel_id)
    rows = df[mask]
    if rows.empty:
        return None

    # Use the same features as training.
    feature_cols = []
    if "Acres" in rows.columns:
        feature_cols.append("Acres")
    if "has_last_sale" in rows.columns:
        feature_cols.append("has_last_sale")
    if "Land Use / Use Code" in rows.columns:
        feature_cols.append("Land Use / Use Code")

    if not feature_cols:
        return None

    X = rows[feature_cols]
    try:
        preds = pipe.predict(X)
    except Exception:
        return None
    if preds.size == 0:
        return None
    return float(preds[0])

