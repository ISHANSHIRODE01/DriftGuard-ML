"""
Baseline ML Model - Student Performance Prediction
===================================================
Production-ready baseline model using Random Forest Regressor.

Model Choice: Random Forest Regressor
- Strong baseline for regression tasks
- Handles non-linear relationships
- Robust to outliers and missing values
- Provides feature importance
- No need for feature scaling (tree-based)
- Good interpretability vs performance trade-off
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    explained_variance_score,
    max_error,
    median_absolute_error
)
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


class BaselineModelTrainer:
    """
    Baseline model trainer for student performance prediction.
    
    Model Selection Rationale:
    ==========================
    
    PRIMARY MODEL: Random Forest Regressor
    
    Why Random Forest:
    ------------------
    1. STRONG BASELINE: Consistently performs well on tabular data
    2. NON-LINEAR: Captures complex relationships between features
    3. ROBUST: Handles outliers and noisy data well
    4. NO SCALING NEEDED: Tree-based, works with raw features
    5. FEATURE IMPORTANCE: Provides interpretability
    6. ENSEMBLE: Reduces overfitting through averaging
    7. PROVEN: Industry standard for baseline models
    
    Why NOT Logistic Regression:
    ----------------------------
    - Logistic Regression is for CLASSIFICATION (binary/multi-class)
    - Our task is REGRESSION (predicting continuous grades 0-20)
    - Would need to convert to classification (lose information)
    
    Alternative: Ridge Regression
    -----------------------------
    - Good linear baseline
    - Fast training
    - Interpretable coefficients
    - BUT: Assumes linear relationships (limiting)
    - We'll train both for comparison
    
    Why NOT Deep Learning:
    ---------------------
    - Small dataset (266 training samples)
    - Risk of overfitting
    - Harder to interpret
    - Longer training time
    - Random Forest likely sufficient
    """
    
    def __init__(self, random_state=42):
        """
        Initialize baseline model trainer.
        
        Args:
            random_state: Seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.feature_names = None
        
    def load_preprocessed_data(self):
        """
        Load preprocessed data from preprocessing pipeline.
        
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        print("=" * 80)
        print("LOADING PREPROCESSED DATA")
        print("=" * 80)
        
        X_train = np.load('models/X_train.npy')
        X_val = np.load('models/X_val.npy')
        X_test = np.load('models/X_test.npy')
        y_train = np.load('models/y_train.npy')
        y_val = np.load('models/y_val.npy')
        y_test = np.load('models/y_test.npy')
        
        # Load feature names
        with open('models/feature_names.txt', 'r') as f:
            self.feature_names = [line.strip().split('. ', 1)[1] for line in f.readlines()]
        
        print(f"\n[OK] Data loaded successfully")
        print(f"  Train: {X_train.shape[0]} samples x {X_train.shape[1]} features")
        print(f"  Validation: {X_val.shape[0]} samples x {X_val.shape[1]} features")
        print(f"  Test: {X_test.shape[0]} samples x {X_test.shape[1]} features")
        print(f"  Feature names: {len(self.feature_names)} features\n")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_random_forest_pipeline(self):
        """
        Create Random Forest pipeline.
        
        HYPERPARAMETER CHOICES:
        =======================
        
        n_estimators=100:
        - Number of trees in the forest
        - 100 is good balance (more trees = better but slower)
        - Diminishing returns after 100-200 trees
        
        max_depth=10:
        - Maximum depth of each tree
        - Prevents overfitting on small dataset
        - None would allow unlimited depth (risk overfitting)
        
        min_samples_split=5:
        - Minimum samples required to split a node
        - Higher value prevents overfitting
        - 5 is reasonable for 266 training samples
        
        min_samples_leaf=2:
        - Minimum samples required at leaf node
        - Prevents creating leaves with single samples
        - Improves generalization
        
        random_state=42:
        - Ensures reproducibility
        - Same results across runs
        
        n_jobs=-1:
        - Use all CPU cores
        - Faster training
        
        Returns:
            sklearn Pipeline with Random Forest
        """
        rf_model = RandomForestRegressor(
            n_estimators=100,      # Number of trees
            max_depth=10,          # Prevent overfitting
            min_samples_split=5,   # Minimum samples to split
            min_samples_leaf=2,    # Minimum samples at leaf
            random_state=self.random_state,
            n_jobs=-1,             # Use all cores
            verbose=0
        )
        
        # Create pipeline (just model for now, preprocessing already done)
        pipeline = Pipeline([
            ('regressor', rf_model)
        ])
        
        return pipeline
    
    def create_ridge_pipeline(self):
        """
        Create Ridge Regression pipeline for comparison.
        
        HYPERPARAMETER CHOICES:
        =======================
        
        alpha=1.0:
        - Regularization strength
        - Higher = more regularization (simpler model)
        - 1.0 is default, good starting point
        
        Returns:
            sklearn Pipeline with Ridge Regression
        """
        ridge_model = Ridge(
            alpha=1.0,
            random_state=self.random_state
        )
        
        pipeline = Pipeline([
            ('regressor', ridge_model)
        ])
        
        return pipeline
    
    def train_model(self, pipeline, X_train, y_train, model_name):
        """
        Train a model pipeline.
        
        Args:
            pipeline: sklearn Pipeline
            X_train: Training features
            y_train: Training targets
            model_name: Name for logging
            
        Returns:
            Trained pipeline
        """
        print(f"\nTraining {model_name}...")
        start_time = datetime.now()
        
        pipeline.fit(X_train, y_train)
        
        training_time = (datetime.now() - start_time).total_seconds()
        print(f"  [OK] Training completed in {training_time:.2f} seconds")
        
        return pipeline, training_time
    
    def evaluate_model(self, pipeline, X, y, dataset_name):
        """
        Comprehensive model evaluation.
        
        METRICS EXPLAINED:
        ==================
        
        1. MAE (Mean Absolute Error):
           - Average absolute difference between predicted and actual
           - In grade points (0-20 scale)
           - Lower is better
           - Most interpretable metric
        
        2. RMSE (Root Mean Squared Error):
           - Square root of average squared errors
           - Penalizes large errors more than MAE
           - Same units as target (grade points)
           - Lower is better
        
        3. R² Score:
           - Proportion of variance explained (0 to 1)
           - 1.0 = perfect predictions
           - 0.0 = predicts mean only
           - Higher is better
           - Standard metric for regression
        
        4. Explained Variance:
           - Similar to R² but different calculation
           - Measures variance explained by model
           - Higher is better
        
        5. Max Error:
           - Worst-case prediction error
           - Identifies outliers
           - Lower is better
        
        6. Median Absolute Error:
           - Robust alternative to MAE
           - Less sensitive to outliers
           - Lower is better
        
        BUSINESS METRICS:
        =================
        
        7. Accuracy within ±1 grade:
           - % of predictions within 1 point of actual
           - Practical measure of "good enough"
           - Higher is better (target: >70%)
        
        8. Accuracy within ±2 grades:
           - % of predictions within 2 points
           - More lenient threshold
           - Higher is better (target: >90%)
        
        Args:
            pipeline: Trained model
            X: Features
            y: True targets
            dataset_name: Name for logging
            
        Returns:
            Dictionary of metrics
        """
        # Make predictions
        y_pred = pipeline.predict(X)
        
        # Calculate metrics
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        explained_var = explained_variance_score(y, y_pred)
        max_err = max_error(y, y_pred)
        median_ae = median_absolute_error(y, y_pred)
        
        # Business metrics
        within_1 = np.mean(np.abs(y - y_pred) <= 1) * 100
        within_2 = np.mean(np.abs(y - y_pred) <= 2) * 100
        
        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'Explained_Variance': explained_var,
            'Max_Error': max_err,
            'Median_AE': median_ae,
            'Accuracy_within_1': within_1,
            'Accuracy_within_2': within_2,
            'predictions': y_pred
        }
        
        return metrics
    
    def print_metrics(self, metrics, model_name, dataset_name):
        """
        Print metrics in a clear, formatted way.
        
        Args:
            metrics: Dictionary of metrics
            model_name: Name of model
            dataset_name: Name of dataset
        """
        print(f"\n{model_name} - {dataset_name} Set Performance:")
        print("-" * 80)
        print(f"  PRIMARY METRICS:")
        print(f"    MAE (Mean Absolute Error):        {metrics['MAE']:.3f} grade points")
        print(f"    RMSE (Root Mean Squared Error):   {metrics['RMSE']:.3f} grade points")
        print(f"    R² Score:                         {metrics['R2']:.4f}")
        
        print(f"\n  SECONDARY METRICS:")
        print(f"    Explained Variance:               {metrics['Explained_Variance']:.4f}")
        print(f"    Median Absolute Error:            {metrics['Median_AE']:.3f} grade points")
        print(f"    Max Error (worst case):           {metrics['Max_Error']:.3f} grade points")
        
        print(f"\n  BUSINESS METRICS:")
        print(f"    Accuracy within ±1 grade:         {metrics['Accuracy_within_1']:.1f}%")
        print(f"    Accuracy within ±2 grades:        {metrics['Accuracy_within_2']:.1f}%")
    
    def compare_models(self, results):
        """
        Compare multiple models side-by-side.
        
        Args:
            results: Dictionary of model results
        """
        print("\n" + "=" * 80)
        print("MODEL COMPARISON (Validation Set)")
        print("=" * 80)
        
        comparison_df = pd.DataFrame({
            model_name: {
                'MAE': result['val_metrics']['MAE'],
                'RMSE': result['val_metrics']['RMSE'],
                'R²': result['val_metrics']['R2'],
                'Acc ±1': result['val_metrics']['Accuracy_within_1'],
                'Acc ±2': result['val_metrics']['Accuracy_within_2'],
                'Training Time (s)': result['training_time']
            }
            for model_name, result in results.items()
        }).T
        
        print("\n" + comparison_df.to_string())
        
        # Determine best model
        best_model = comparison_df['R²'].idxmax()
        print(f"\n[BEST MODEL]: {best_model} (Highest R² = {comparison_df.loc[best_model, 'R²']:.4f})")
        
        return comparison_df
    
    def get_feature_importance(self, pipeline, top_n=15):
        """
        Extract and display feature importance from Random Forest.
        
        Args:
            pipeline: Trained Random Forest pipeline
            top_n: Number of top features to show
            
        Returns:
            DataFrame of feature importances
        """
        if 'RandomForest' not in str(type(pipeline.named_steps['regressor'])):
            print("  [SKIP] Feature importance only available for Random Forest")
            return None
        
        # Get feature importances
        importances = pipeline.named_steps['regressor'].feature_importances_
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        print(f"\n  TOP {top_n} MOST IMPORTANT FEATURES:")
        print("  " + "-" * 76)
        for idx, row in importance_df.head(top_n).iterrows():
            print(f"    {row['Feature']:30s}: {row['Importance']:.4f}")
        
        return importance_df
    
    def save_model(self, pipeline, model_name):
        """
        Save trained model for production use.
        
        Args:
            pipeline: Trained pipeline
            model_name: Name for the saved file
        """
        filepath = f'models/{model_name}_baseline.pkl'
        joblib.dump(pipeline, filepath)
        print(f"\n[OK] Model saved to: {filepath}")
    
    def run_baseline_training(self):
        """
        Complete baseline model training workflow.
        """
        print("=" * 80)
        print("BASELINE MODEL TRAINING - STUDENT PERFORMANCE PREDICTION")
        print("=" * 80)
        
        # Load data
        X_train, X_val, X_test, y_train, y_val, y_test = self.load_preprocessed_data()
        
        # Train Random Forest
        print("\n" + "=" * 80)
        print("MODEL 1: RANDOM FOREST REGRESSOR (PRIMARY BASELINE)")
        print("=" * 80)
        print("\nWhy Random Forest:")
        print("  - Strong baseline for tabular data")
        print("  - Handles non-linear relationships")
        print("  - Robust to outliers")
        print("  - Provides feature importance")
        print("  - No scaling needed (tree-based)")
        
        rf_pipeline = self.create_random_forest_pipeline()
        rf_pipeline, rf_time = self.train_model(rf_pipeline, X_train, y_train, "Random Forest")
        
        # Evaluate on train and validation
        rf_train_metrics = self.evaluate_model(rf_pipeline, X_train, y_train, "Training")
        rf_val_metrics = self.evaluate_model(rf_pipeline, X_val, y_val, "Validation")
        
        self.print_metrics(rf_train_metrics, "Random Forest", "Training")
        self.print_metrics(rf_val_metrics, "Random Forest", "Validation")
        
        # Feature importance
        print("\n" + "-" * 80)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("-" * 80)
        rf_importance = self.get_feature_importance(rf_pipeline)
        
        # Store results
        self.models['Random Forest'] = rf_pipeline
        self.results['Random Forest'] = {
            'train_metrics': rf_train_metrics,
            'val_metrics': rf_val_metrics,
            'training_time': rf_time,
            'feature_importance': rf_importance
        }
        
        # Train Ridge Regression for comparison
        print("\n" + "=" * 80)
        print("MODEL 2: RIDGE REGRESSION (LINEAR BASELINE)")
        print("=" * 80)
        print("\nWhy Ridge Regression:")
        print("  - Simple linear baseline")
        print("  - Fast training")
        print("  - Interpretable coefficients")
        print("  - Good for comparison")
        
        ridge_pipeline = self.create_ridge_pipeline()
        ridge_pipeline, ridge_time = self.train_model(ridge_pipeline, X_train, y_train, "Ridge Regression")
        
        # Evaluate
        ridge_train_metrics = self.evaluate_model(ridge_pipeline, X_train, y_train, "Training")
        ridge_val_metrics = self.evaluate_model(ridge_pipeline, X_val, y_val, "Validation")
        
        self.print_metrics(ridge_train_metrics, "Ridge Regression", "Training")
        self.print_metrics(ridge_val_metrics, "Ridge Regression", "Validation")
        
        # Store results
        self.models['Ridge Regression'] = ridge_pipeline
        self.results['Ridge Regression'] = {
            'train_metrics': ridge_train_metrics,
            'val_metrics': ridge_val_metrics,
            'training_time': ridge_time
        }
        
        # Compare models
        comparison_df = self.compare_models(self.results)
        
        # Save best model
        print("\n" + "=" * 80)
        print("SAVING MODELS")
        print("=" * 80)
        self.save_model(self.models['Random Forest'], 'random_forest')
        self.save_model(self.models['Ridge Regression'], 'ridge')
        
        # Final summary
        print("\n" + "=" * 80)
        print("BASELINE TRAINING COMPLETE!")
        print("=" * 80)
        
        print("\nKEY FINDINGS:")
        print("-" * 80)
        rf_val = self.results['Random Forest']['val_metrics']
        ridge_val = self.results['Ridge Regression']['val_metrics']
        
        print(f"  Random Forest:")
        print(f"    - MAE: {rf_val['MAE']:.3f} grade points")
        print(f"    - R²: {rf_val['R2']:.4f}")
        print(f"    - Accuracy ±1: {rf_val['Accuracy_within_1']:.1f}%")
        
        print(f"\n  Ridge Regression:")
        print(f"    - MAE: {ridge_val['MAE']:.3f} grade points")
        print(f"    - R²: {ridge_val['R2']:.4f}")
        print(f"    - Accuracy ±1: {ridge_val['Accuracy_within_1']:.1f}%")
        
        # Interpretation
        print("\nINTERPRETATION:")
        print("-" * 80)
        if rf_val['MAE'] < 2.0:
            print("  [EXCELLENT] MAE < 2.0: Predictions within 2 grade points on average")
        elif rf_val['MAE'] < 3.0:
            print("  [GOOD] MAE < 3.0: Reasonable prediction accuracy")
        else:
            print("  [FAIR] MAE >= 3.0: Room for improvement")
        
        if rf_val['R2'] > 0.80:
            print("  [EXCELLENT] R² > 0.80: Model explains >80% of variance")
        elif rf_val['R2'] > 0.70:
            print("  [GOOD] R² > 0.70: Model explains >70% of variance")
        else:
            print("  [FAIR] R² < 0.70: Consider advanced models")
        
        if rf_val['Accuracy_within_1'] > 70:
            print("  [EXCELLENT] >70% predictions within ±1 grade point")
        elif rf_val['Accuracy_within_1'] > 60:
            print("  [GOOD] >60% predictions within ±1 grade point")
        else:
            print("  [FAIR] <60% predictions within ±1 grade point")
        
        print("\nNEXT STEPS:")
        print("-" * 80)
        print("  1. Evaluate best model on test set (final evaluation)")
        print("  2. Analyze prediction errors and residuals")
        print("  3. Consider hyperparameter tuning (Grid Search)")
        print("  4. Try advanced models (XGBoost, LightGBM)")
        print("  5. Deploy best model to production")
        print()
        
        return self.models, self.results, comparison_df


def main():
    """Main execution function."""
    trainer = BaselineModelTrainer(random_state=42)
    models, results, comparison = trainer.run_baseline_training()
    return trainer, models, results, comparison


if __name__ == "__main__":
    trainer, models, results, comparison = main()
