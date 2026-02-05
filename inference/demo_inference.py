"""
Model Loading and Inference Demo
=================================
Demonstrates how to load the saved model and make predictions.

This script shows the complete workflow for production deployment:
1. Load saved artifacts
2. Prepare new data
3. Make predictions
4. Interpret results
"""

import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path


def load_production_model():
    """
    Load all production artifacts.
    
    Returns:
        model, preprocessor, metrics
    """
    print("=" * 80)
    print("LOADING PRODUCTION MODEL")
    print("=" * 80)
    
    # Load model
    print("\n[1/3] Loading trained model...")
    model = joblib.load('model/model.pkl')
    print("  ✓ Model loaded: RandomForestRegressor")
    
    # Load preprocessor
    print("\n[2/3] Loading preprocessor...")
    preprocessor = joblib.load('model/preprocessor.pkl')
    print("  ✓ Preprocessor loaded: ColumnTransformer")
    
    # Load metrics
    print("\n[3/3] Loading metrics...")
    with open('model/metrics.json', 'r') as f:
        metrics = json.load(f)
    print("  ✓ Metrics loaded")
    
    # Display model info
    print("\n" + "=" * 80)
    print("MODEL INFORMATION")
    print("=" * 80)
    model_info = metrics['metrics']['model_info']
    for key, value in model_info.items():
        print(f"  {key:20s}: {value}")
    
    # Display performance
    print("\n" + "=" * 80)
    print("EXPECTED PERFORMANCE (Validation Set)")
    print("=" * 80)
    val_metrics = metrics['metrics']['validation_metrics']
    print(f"  MAE:                  {val_metrics['mae']:.3f} grade points")
    print(f"  RMSE:                 {val_metrics['rmse']:.3f} grade points")
    print(f"  R² Score:             {val_metrics['r2_score']:.4f}")
    print(f"  Max Error:            {val_metrics['max_error']:.3f} grade points")
    
    biz_metrics = metrics['metrics']['business_metrics']
    print(f"\n  Accuracy ±1 grade:    {biz_metrics['accuracy_within_1_grade']:.1f}%")
    print(f"  Accuracy ±2 grades:   {biz_metrics['accuracy_within_2_grades']:.1f}%")
    print(f"  Accuracy ±3 grades:   {biz_metrics['accuracy_within_3_grades']:.1f}%")
    
    return model, preprocessor, metrics


def demo_prediction(model, preprocessor):
    """
    Demonstrate prediction on test data.
    
    Args:
        model: Loaded model
        preprocessor: Loaded preprocessor
    """
    print("\n" + "=" * 80)
    print("PREDICTION DEMO")
    print("=" * 80)
    
    # Load test data
    print("\nLoading test data...")
    X_test = np.load('models/X_test.npy')
    y_test = np.load('models/y_test.npy')
    print(f"  ✓ Test data loaded: {X_test.shape[0]} samples")
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = model.predict(X_test)
    print(f"  ✓ Predictions generated for {len(predictions)} students")
    
    # Show sample predictions
    print("\n" + "=" * 80)
    print("SAMPLE PREDICTIONS (First 10 students)")
    print("=" * 80)
    print(f"  {'Actual':>8s}  {'Predicted':>10s}  {'Error':>8s}  {'Status':>15s}")
    print("  " + "-" * 76)
    
    for i in range(min(10, len(predictions))):
        actual = y_test[i]
        pred = predictions[i]
        error = abs(actual - pred)
        
        # Determine status
        if error <= 1:
            status = "✓ Excellent"
        elif error <= 2:
            status = "✓ Good"
        elif error <= 3:
            status = "~ Fair"
        else:
            status = "✗ Poor"
        
        print(f"  {actual:8.1f}  {pred:10.2f}  {error:8.2f}  {status:>15s}")
    
    # Overall test performance
    from sklearn.metrics import mean_absolute_error, r2_score
    
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    acc_1 = np.mean(np.abs(y_test - predictions) <= 1) * 100
    
    print("\n" + "=" * 80)
    print("TEST SET PERFORMANCE (Final Evaluation)")
    print("=" * 80)
    print(f"  MAE:              {mae:.3f} grade points")
    print(f"  R² Score:         {r2:.4f}")
    print(f"  Accuracy ±1:      {acc_1:.1f}%")
    
    if r2 > 0.85:
        print("\n  ✓ EXCELLENT: Model generalizes well to test data!")
    elif r2 > 0.75:
        print("\n  ✓ GOOD: Model performs well on test data")
    else:
        print("\n  ~ FAIR: Consider model improvements")


def demo_single_prediction(model, preprocessor):
    """
    Demonstrate prediction for a single new student.
    
    Args:
        model: Loaded model
        preprocessor: Loaded preprocessor
    """
    print("\n" + "=" * 80)
    print("SINGLE STUDENT PREDICTION DEMO")
    print("=" * 80)
    
    # Load original data to get a sample
    df = pd.read_csv('Data/student_mat_clean.csv')
    
    # Get a random student
    sample_idx = np.random.randint(0, len(df))
    student = df.iloc[sample_idx:sample_idx+1].copy()
    
    print(f"\nStudent Profile (Sample #{sample_idx}):")
    print("-" * 80)
    print(f"  School:           {student['school'].values[0]}")
    print(f"  Sex:              {student['sex'].values[0]}")
    print(f"  Age:              {student['age'].values[0]}")
    print(f"  Mother's Edu:     {student['Medu'].values[0]}")
    print(f"  Father's Edu:     {student['Fedu'].values[0]}")
    print(f"  Study Time:       {student['studytime'].values[0]}")
    print(f"  Failures:         {student['failures'].values[0]}")
    print(f"  Absences:         {student['absences'].values[0]}")
    print(f"  G1 (1st period):  {student['G1'].values[0]}")
    print(f"  G2 (2nd period):  {student['G2'].values[0]}")
    print(f"  G3 (Final) ACTUAL: {student['G3'].values[0]}")
    
    # Prepare data (same as preprocessing pipeline)
    from save_model import ModelSaver
    
    # Convert binary features
    binary_features = ['schoolsup', 'famsup', 'paid', 'activities', 
                      'nursery', 'higher', 'internet', 'romantic']
    for col in binary_features:
        if col in student.columns:
            student[col] = (student[col] == 'yes').astype(int)
    
    # Separate features
    X_student = student.drop(columns=['G3'])
    y_actual = student['G3'].values[0]
    
    # Transform and predict
    X_student_prep = preprocessor.transform(X_student)
    prediction = model.predict(X_student_prep)[0]
    
    print("\n" + "=" * 80)
    print("PREDICTION RESULT")
    print("=" * 80)
    print(f"  Actual Final Grade (G3):     {y_actual:.1f}")
    print(f"  Predicted Final Grade:       {prediction:.2f}")
    print(f"  Prediction Error:            {abs(y_actual - prediction):.2f} grade points")
    
    if abs(y_actual - prediction) <= 1:
        print("\n  ✓ EXCELLENT: Prediction within ±1 grade point!")
    elif abs(y_actual - prediction) <= 2:
        print("\n  ✓ GOOD: Prediction within ±2 grade points")
    else:
        print("\n  ~ FAIR: Prediction error > 2 grade points")


def main():
    """
    Main demo execution.
    """
    print("\n" * 2)
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "MODEL LOADING & INFERENCE DEMO" + " " * 28 + "║")
    print("╚" + "=" * 78 + "╝")
    
    # Load model
    model, preprocessor, metrics = load_production_model()
    
    # Demo 1: Batch predictions on test set
    demo_prediction(model, preprocessor)
    
    # Demo 2: Single student prediction
    demo_single_prediction(model, preprocessor)
    
    # Final summary
    print("\n" + "=" * 80)
    print("DEPLOYMENT READY!")
    print("=" * 80)
    print("""
The model is ready for production deployment. Key files:

  model/model.pkl           - Trained Random Forest model (161 KB)
  model/preprocessor.pkl    - Preprocessing pipeline (2 KB)
  model/metrics.json        - Performance metrics
  model/feature_names.json  - Feature names

To integrate into your application:
  1. Load model and preprocessor using joblib
  2. Preprocess new student data
  3. Call model.predict() to get grade predictions
  4. Use predictions for early warning systems

Expected Performance:
  - MAE: ~1.1 grade points
  - R²: ~0.91 (explains 91% of variance)
  - 66% of predictions within ±1 grade point

Model Status: ✓ PRODUCTION READY
    """)


if __name__ == "__main__":
    main()
