import sys
print(sys.path)
try:
    import evidently
    print(f"Evidently file: {evidently.__file__}")
    print(f"Evidently version: {evidently.__version__}")
    print(f"Dir: {dir(evidently)}")
    
    import evidently.report
    print("Report module found")
except Exception as e:
    print(f"Error: {e}")
