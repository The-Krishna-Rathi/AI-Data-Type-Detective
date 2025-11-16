import json
from typing import Dict, Any
from openai import OpenAI
import ollama

class AIInferenceEngine:

    def __init__(self, model='llama3.2'):
        self.client = OpenAI()
        self.model = model
    
    def infer_column_types(self, column_name: str, profile: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"""
        You are a data type inference expert. Infer the BEST semantic data type for this column.

        Allowed types:
        - string
        - integer
        - float
        - boolean
        - date
        - currency
        - categorical
        - unknown

        Provide reasoning + confidence.

        Column Name: {column_name}
        Profiling Data:
        {json.dumps(profile, indent=4)}

        Respond only in this JSON structure:
        {{
            "column_name": "{column_name}",
            "inferred_type": "",
            "confidence": 0.0,
            "reasoning": ""
        }}
        """

        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )

        # Extract AI output
        ai_text = response["message"]["content"]

        # Parse JSON safely
        try:
            return json.loads(ai_text)
        except Exception:
            return {
                "column_name": column_name,
                "inferred_type": "unknown",
                "confidence": 0.0,
                "reasoning": "Failed to parse AI response."
            }
    
    def infer_dataframe(self, profiling_report: Dict[str, Any]):
        results = {}

        for col, profile in profiling_report.items():
            print(f"Inferring AI type for: {col}")
            result = self.infer_column_types(col, profile)
            results[col] = result

        return results