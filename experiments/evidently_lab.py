"""
Evidently AI Drift Detection Integration
=========================================
Generates a comprehensive Data Drift report using Evidently AI.
Saves the report as an interactive HTML file.

Usage:
    python evidently_integration.py

Requirements:
    pip install evidently==0.4.15
"""

import pandas as pd
import json

# Standard imports for Evidently 0.4.x
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def generate_evidently_report(
    reference_path: str = 'data/train.csv',
    current_path: str = 'data/new_data.csv',
    output_path: str = 'data/drift_report_evidently.html'
):
    """
    Generate an Evidently AI Data Drift report.
    
    Args:
        reference_path: Path to reference (training) data
        current_path: Path to current (production) data
        output_path: Path to save the HTML report
    """
    print("=" * 80)
    print("EVIDENTLY AI DRIFT DETECTION (v0.4.15)")
    print("=" * 80)
    
    # 1. Load Data
    print(f"\n[1/3] Loading data...")
    reference_df = pd.read_csv(reference_path)
    current_df = pd.read_csv(current_path)
    
    print(f"  Reference data: {reference_df.shape[0]} rows")
    print(f"  Current data:   {current_df.shape[0]} rows")
    
    # 2. Build Report
    print(f"\n[2/3] Generating report (this may take a moment)...")
    
    report = Report(metrics=[
        DataDriftPreset(),      # Comprehensive drift preset
    ])
    
    report.run(reference_data=reference_df, current_data=current_df)
    
    # 3. Save Report
    print(f"\n[3/3] Saving report to {output_path}...")
    report.save_html(output_path)
    
    # Extract summary metrics from the JSON representation
    try:
        json_summary = json.loads(report.json())
        
        # Structure for 0.4.x
        drift_share = json_summary['metrics'][0]['result']['drift_share']
        number_of_drifted_features = json_summary['metrics'][0]['result']['number_of_drifted_features']
        dataset_drift = json_summary['metrics'][0]['result']['dataset_drift']
        
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"  Dataset Drift Detected: {'YES ✗' if dataset_drift else 'NO ✓'}")
        print(f"  Drifted Features:       {number_of_drifted_features}")
        print(f"  Drift Share:            {drift_share * 100:.1f}%")
        print(f"  Saved to:               {output_path}")
        print("\n  Open the HTML report for detailed visualizations!")
        
    except Exception as e:
        print(f"\n  Report generated but summary extraction skipped: {e}")
        print(f"  Saved to: {output_path}")
        
    print("\n" + "=" * 80)

if __name__ == "__main__":
    generate_evidently_report()
