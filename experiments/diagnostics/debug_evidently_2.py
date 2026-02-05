import evidently
print(f"Type of evidently.Report: {type(evidently.Report)}")
try:
    from evidently.report import Report
    print("Success: from evidently.report import Report")
except Exception as e:
    print(f"Fail: {e}")

try:
    from evidently import Report
    print("Success: from evidently import Report")
except Exception as e:
    print(f"Fail: {e}")
