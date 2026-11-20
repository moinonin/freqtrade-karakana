"""Tabular preprocessing and text/categorical vectorization for Karakana SGO trainers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from karakana_engine.trainers.dataset import PREDICTION_OUTPUT_COLUMNS, is_exported_index_column
from sklearn.feature_extraction.text import TfidfVectorizer


class TabularVectorizer:
    """
    Transforms mixed tabular data (raw text, categoricals, numerics) into dense numeric matrices.
    Designed for Karakana's SGO DatasetTrainer.
    """

    def __init__(
        self,
        max_text_features: int = 48,
        max_category_cardinality: int = 50,
        stop_words: str = "english",
    ):
        self.max_text_features = max_text_features
        self.max_category_cardinality = max_category_cardinality
        self.stop_words = stop_words

        self.numeric_cols: list[str] = []
        self.categorical_cols: list[str] = []
        self.categorical_levels: dict[str, list[str]] = {}
        self.text_cols: list[str] = []
        self.text_vectorizers: dict[str, TfidfVectorizer] = {}
        self.feature_columns_: list[str] = []
        self.is_fitted: bool = False

    def fit_transform(
        self,
        frame: pd.DataFrame,
        target_col: str,
        ignored_cols: list[str] | None = None,
    ) -> pd.DataFrame:
        """Fit vectorizer on training DataFrame and return numeric features + target column."""
        ignored = set(ignored_cols or [])
        df = frame.copy()

        self.numeric_cols = []
        self.categorical_cols = []
        self.categorical_levels = {}
        self.text_cols = []
        self.text_vectorizers = {}

        candidate_cols = [c for c in df.columns if c != target_col and c not in ignored]

        for col in candidate_cols:
            if is_exported_index_column(col) or col.strip().lower() in PREDICTION_OUTPUT_COLUMNS:
                continue

            series = df[col]
            if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
                self.numeric_cols.append(col)
            elif pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
                non_null = series.dropna().astype(str)
                if non_null.empty:
                    continue
                unique_count = non_null.nunique()
                avg_word_count = non_null.apply(lambda s: len(s.split())).mean()

                if unique_count <= self.max_category_cardinality and avg_word_count < 4:
                    self.categorical_cols.append(col)
                    levels = sorted(non_null.unique().tolist())
                    self.categorical_levels[col] = levels
                else:
                    self.text_cols.append(col)
                    vec = TfidfVectorizer(
                        max_features=self.max_text_features,
                        stop_words=self.stop_words,
                    )
                    vec.fit(non_null)
                    self.text_vectorizers[col] = vec

        self.is_fitted = True
        return self._transform_frame(df, target_col=target_col)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Transform inference records into the fitted numeric feature representation."""
        if not self.is_fitted:
            raise RuntimeError("TabularVectorizer must be fitted before calling transform")
        return self._transform_frame(frame, target_col=None)

    def _transform_frame(self, df: pd.DataFrame, target_col: str | None = None) -> pd.DataFrame:
        feature_parts: list[pd.DataFrame] = []

        # 1. Numeric columns
        if self.numeric_cols:
            num_df = df[self.numeric_cols].copy()
            for col in self.numeric_cols:
                num_df[col] = pd.to_numeric(num_df[col], errors="coerce").fillna(0.0)
            feature_parts.append(num_df)

        # 2. Categorical columns (one-hot)
        for col in self.categorical_cols:
            series = df[col].astype(str).fillna("missing")
            levels = self.categorical_levels[col]
            dummy_dict = {}
            for lvl in levels:
                dummy_dict[f"{col}_{lvl}"] = (series == lvl).astype(float)
            cat_df = pd.DataFrame(dummy_dict, index=df.index)
            feature_parts.append(cat_df)

        # 3. Free-text columns (TF-IDF)
        for col in self.text_cols:
            vec = self.text_vectorizers[col]
            text_series = df[col].fillna("").astype(str)
            tfidf_mat = vec.transform(text_series).toarray()
            feat_names = [f"{col}_tfidf_{word}" for word in vec.get_feature_names_out()]
            txt_df = pd.DataFrame(tfidf_mat, columns=feat_names, index=df.index)
            feature_parts.append(txt_df)

        if not feature_parts:
            raise ValueError("No usable features extracted from DataFrame")

        result = pd.concat(feature_parts, axis=1)
        if target_col and target_col in df.columns:
            result[target_col] = df[target_col].values

        if not self.feature_columns_:
            self.feature_columns_ = [c for c in result.columns if c != target_col]

        return result

    def to_dict(self) -> dict[str, Any]:
        """Serialize fitted vectorizer metadata for inference."""
        if not self.is_fitted:
            return {}
        text_data = {}
        for col, vec in self.text_vectorizers.items():
            text_data[col] = {
                "vocabulary": {str(k): int(v) for k, v in vec.vocabulary_.items()},
                "idf": vec.idf_.tolist(),
            }
        return {
            "numeric_cols": list(self.numeric_cols),
            "categorical_cols": list(self.categorical_cols),
            "categorical_levels": {col: list(lvls) for col, lvls in self.categorical_levels.items()},
            "text_cols": list(self.text_cols),
            "text_vectorizers": text_data,
            "feature_columns": list(self.feature_columns_),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TabularVectorizer:
        """Reconstruct fitted vectorizer from serialized dictionary."""
        instance = cls()
        instance.numeric_cols = list(data.get("numeric_cols", []))
        instance.categorical_cols = list(data.get("categorical_cols", []))
        instance.categorical_levels = dict(data.get("categorical_levels", {}))
        instance.text_cols = list(data.get("text_cols", []))
        instance.feature_columns_ = list(data.get("feature_columns", []))
        instance.text_vectorizers = {}
        for col, vec_info in data.get("text_vectorizers", {}).items():
            vec = TfidfVectorizer(vocabulary=vec_info["vocabulary"], stop_words=instance.stop_words)
            vec.idf_ = np.array(vec_info["idf"], dtype=float)
            instance.text_vectorizers[col] = vec
        instance.is_fitted = True
        return instance

    def transform_records(self, records: list[dict[str, Any]]) -> list[dict[str, float]]:
        """Transform raw input records into vectorized feature dictionaries."""
        df = pd.DataFrame(records)
        transformed_df = self.transform(df)
        return transformed_df[self.feature_columns_].to_dict(orient="records")


def prepare_tabular_dataset(
    csv_path: str,
    target_col: str,
    output_path: str | None = None,
    max_text_features: int = 48,
) -> tuple[pd.DataFrame, TabularVectorizer]:
    """
    Read a CSV, strip ID/index columns, vectorize categoricals and text,
    and optionally save to a prepared CSV.
    """
    df = pd.read_csv(csv_path)
    vectorizer = TabularVectorizer(max_text_features=max_text_features)
    vectorized_df = vectorizer.fit_transform(df, target_col=target_col)

    if output_path:
        vectorized_df.to_csv(output_path, index=False)

    return vectorized_df, vectorizer
