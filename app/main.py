import pandas as pd
from profiler.profiler import Profiler
from ai_inference.ai_engine import AIInferenceEngine
import json
from dotenv import load_dotenv

df = pd.read_csv("../data/sample.csv")

load_dotenv()  # loads .env from current or parent dir

profiler = Profiler()
profiling_report = profiler.profile_dataframe(df)

ai_engine = AIInferenceEngine(model='llama3.2')
inference_report = ai_engine.infer_dataframe(profiling_report)

print(json.dumps(inference_report, indent=4))
