import evidently
try:
    from evidently.metric_preset import DataDriftPreset
    print("Success: from evidently.metric_preset import DataDriftPreset")
except:
    print("Fail: from evidently.metric_preset import DataDriftPreset")

try:
    from evidently.presets import DataDriftPreset
    print("Success: from evidently.presets import DataDriftPreset")
except:
    print("Fail: from evidently.presets import DataDriftPreset")
