from evidently import Report
from evidently.presets import DataDriftPreset
import pandas as pd

print(f"Report class: {Report}")
r = Report(metrics=[DataDriftPreset()])
print(f"Report instance keys: {dir(r)}")
