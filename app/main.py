import pandas as pd
from profiler.profiler import Profiler

df = pd.read_csv("../data/sample.csv")

profiler = Profiler(sample_size=5)
report = profiler.profile_dataframe(df)

import json
print(json.dumps(report, indent=4))
