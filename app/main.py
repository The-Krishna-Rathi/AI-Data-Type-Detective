from profiler.profiler import Profiler
from ai_inference.ai_engine import AIInferenceEngine
from rules.rule_engine import RuleEngine
from hybrid.hybrid_classifier import HybridClassifier

from dotenv import load_dotenv
import pandas as pd
import json


df = pd.read_csv("../data/sample.csv")

load_dotenv()  # loads .env from current or parent dir

profiler = Profiler()
rule_engine = RuleEngine()
ai = AIInferenceEngine()

hybrid = HybridClassifier(profiler, rule_engine, ai)

results = {}

for col in df.columns:
    print(f"Classifying column: {col}")
    results[col] = hybrid.classify_column(df[col], col)

print(json.dumps(results, indent=4))
