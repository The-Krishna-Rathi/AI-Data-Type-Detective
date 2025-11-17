import re
import pandas as pd

class RuleEngine:

    def detect(self, series: pd.Series) -> dict:
        col = series.dropna().astype(str)

        # RULE 1 - BOOLEAN
        if col.str.lower().isin(["true", "false", "yes", "no", "0", "1"]).mean() > 0.9:
            return {"type": "boolean", "confidence": 0.95, "reason": "Boolean-like values detected"}

        # RULE 2 - INTEGER
        if col.str.match(r"^-?\d+$").mean() > 0.9:
            return {"type": "integer", "confidence": 0.90, "reason": "Mostly whole numbers"}

        # RULE 3 - FLOAT
        if col.str.match(r"^-?\d+(\.\d+)?$").mean() > 0.9:
            return {"type": "float", "confidence": 0.85, "reason": "Decimal number detected"}

        # RULE 4 - CURRENCY
        currency_pattern = r"^[$₹€£]\s*\d[\d,]*(\.\d+)?$"
        if col.str.match(currency_pattern).mean() > 0.7:
            return {"type": "currency", "confidence": 0.90, "reason": "Currency symbol detected"}

        # RULE 5 - DATETIME
        date_pattern = r"\d{1,4}[-/]\d{1,2}[-/]\d{1,4}"
        if col.str.contains(date_pattern).mean() > 0.7:
            return {"type": "datetime", "confidence": 0.90, "reason": "Date-like pattern detected"}

        # RULE 6 - CATEGORICAL (few unique values)
        if series.nunique() / len(series) < 0.1:
            return {"type": "categorical", "confidence": 0.80, "reason": "Low unique ratio"}

        # RULE 7 - DEFAULT STRING
        return {"type": "string", "confidence": 0.60, "reason": "Fallback: string"}
