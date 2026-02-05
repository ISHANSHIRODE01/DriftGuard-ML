import pandas as pd
import json
from pathlib import Path
from drift.detector import DriftDetector

# Load real data from the new raw directory
train_df = pd.read_csv('data/raw/train.csv')
new_df = pd.read_csv('data/raw/new_data.csv')

# Run detection
detector = DriftDetector()
detector.detect_drift(train_df, new_df)
report = detector.get_drift_report()

# Save JSON to the reports directory
output_path = Path('data/reports/drift_report.json')
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, 'w') as f:
    json.dump(report, f, indent=2)

print(f"✓ Updated {output_path} with new drift detection metrics.")
