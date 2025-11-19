from profiler.profiler import Profiler
from ai_inference.ai_engine import AIInferenceEngine
from rules.rule_engine import RuleEngine
from hybrid.hybrid_classifier import HybridClassifier
from processor.type_convertor import TypeConverter

from dotenv import load_dotenv
import pandas as pd
import json

def main():
    load_dotenv()  # loads .env from current or parent dir

    # Step 1: Load CSV
    df = pd.read_csv("../data/sample.csv")

    # Step 2: Profiler
    profiler = Profiler()

    # Step 3: AI inference
    ai_engine = AIInferenceEngine()

    # Step 4: Rule-based
    rule_engine = RuleEngine()

    # Step 5: Hybrid engine selects final datatype
    print("Running Hybrid Classifier.....")
    hybrid = HybridClassifier(profiler, rule_engine, ai_engine)

    final_schema = {}
    for col in df.columns:
        final_schema[col] = hybrid.classify_column(df[col], col)

    print("\n--- Final Schema ---")
    print(json.dumps(final_schema, indent=4))

    # Step 6: APPLY DATATYPE CONVERSIONS  (Milestone 5)
    print("Running Final type conversion.....")
    converter = TypeConverter()
    cleaned_df, conversion_report = converter.apply_final_datatypes(df, final_schema)

    print("\n---original df----")
    print(df.dtypes)

    print("\n--- Conversion Report ---")
    print(conversion_report)

    print("\n--- Cleaned DataFrame ---")
    print(cleaned_df.dtypes)
    print(cleaned_df.head())

if __name__ == "__main__":
    main()