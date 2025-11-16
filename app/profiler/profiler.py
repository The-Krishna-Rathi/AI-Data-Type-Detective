import pandas as pd
import numpy as np
import re
from typing import Dict, Any

class Profiler:

    def __init__(self, sample_size: int = 5):
        self.sample_size = sample_size

    def _to_py(self, value):
        """Convert numpy types to native Python types for JSON safety."""
        if isinstance(value, (np.integer, np.int64)):
            return int(value)
        if isinstance(value, (np.floating, np.float64)):
            return float(value)
        if isinstance(value, (np.bool_)):
            return bool(value)
        return value

    def _convert_dict(self, d: dict) -> dict:
        """Recursively converts all values in a dict to JSON-friendly types."""
        new_d = {}
        for k, v in d.items():
            if isinstance(v, dict):
                new_d[k] = self._convert_dict(v)
            else:
                new_d[k] = self._to_py(v)
        return new_d

    def profile_column(self, series: pd.Series) -> Dict[str, Any]:
        col = series.dropna().astype(str)

        # Sample values
        samples = col.sample(min(self.sample_size, len(col)), random_state=42).tolist()

        # Basic stats
        null_pct = series.isna().mean() * 100
        unique_count = series.nunique(dropna=True)

        # Length stats
        lengths = col.apply(len)
        min_len = lengths.min() if not lengths.empty else None
        max_len = lengths.max() if not lengths.empty else None
        avg_len = lengths.mean() if not lengths.empty else None

        # Pattern detection
        pattern_info = {
            "has_digits": col.str.contains(r"\d").mean(),
            "has_alpha": col.str.contains(r"[A-Za-z]").mean(),
            "has_special_chars": col.str.contains(r"[^A-Za-z0-9]").mean(),
            "is_possible_date": col.str.contains(r"\d{1,4}[-/]\d{1,2}[-/]\d{1,4}").mean(),
            "is_possible_currency": col.str.contains(r"[$₹€]|,\d{3}").mean(),
            "is_possible_bool": col.str.lower().isin(["true", "false", "yes", "no"]).mean()
        }

        numeric_stats = None
        if pd.api.types.is_numeric_dtype(series):
            numeric_stats = {
                "mean": series.mean(),
                "min": series.min(),
                "max": series.max(),
                "std": series.std(),
            }

        result = {
            "dtype": str(series.dtype),
            "null_percentage": null_pct,
            "unique_count": unique_count,
            "samples": samples,
            "min_length": min_len,
            "max_length": max_len,
            "avg_length": avg_len,
            "pattern_info": pattern_info,
            "numeric_stats": numeric_stats
        }

        # Convert everything into JSON-friendly types
        return self._convert_dict(result)

    def profile_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        report = {}
        for col in df.columns:
            print(f"Profiling column: {col}")
            report[col] = self.profile_column(df[col])

        return report
