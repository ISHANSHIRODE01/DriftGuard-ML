# DATA DRIFT - INTERVIEW PREPARATION GUIDE
# =========================================

"""
WHAT IS DATA DRIFT?
===================

Data drift (also called dataset shift or covariate shift) occurs when the statistical 
properties of the input features change over time between training and production data.

INTERVIEW ANSWER (30-second version):
-------------------------------------
"Data drift is when the distribution of input features in production differs from 
the training data. This causes model performance to degrade because the model was 
trained on different data patterns. We detect it using statistical tests like 
Kolmogorov-Smirnov test and monitor key metrics to trigger model retraining."


DETAILED EXPLANATION FOR INTERVIEWS:
====================================

1. TYPES OF DRIFT:
------------------

a) COVARIATE DRIFT (Feature Drift):
   - Definition: Distribution of input features (X) changes
   - Example: In student performance model, if average age shifts from 16 to 18
   - Impact: Model sees data it wasn't trained on
   - Detection: KS-test, PSI (Population Stability Index)

b) PRIOR PROBABILITY DRIFT (Label Drift):
   - Definition: Distribution of target variable (Y) changes
   - Example: Pass rate drops from 80% to 60%
   - Impact: Model's predictions become biased
   - Detection: Chi-square test, comparing proportions

c) CONCEPT DRIFT:
   - Definition: Relationship between X and Y changes
   - Example: Study time used to predict grades well, but now doesn't
   - Impact: Model's learned patterns become invalid
   - Detection: Model performance monitoring (accuracy, MAE drop)


2. WHY DOES DRIFT HAPPEN?
--------------------------

Real-world examples:
- Seasonality: Student behavior changes during exam season
- Policy changes: New grading system implemented
- Population shift: Different student demographics
- Data collection changes: New sensors, different survey questions
- External events: Pandemic affecting attendance patterns


3. WHY IS DRIFT DETECTION IMPORTANT?
-------------------------------------

Without drift detection:
✗ Model performance silently degrades
✗ Business decisions based on wrong predictions
✗ No trigger for model retraining
✗ Loss of trust in ML system

With drift detection:
✓ Early warning system for model degradation
✓ Automated retraining triggers
✓ Maintain model performance over time
✓ Build trust through monitoring


4. HOW TO DETECT DRIFT?
------------------------

Statistical Tests:
------------------

a) Kolmogorov-Smirnov (KS) Test:
   - For: Numerical features
   - Tests: Whether two samples come from same distribution
   - Output: p-value (< 0.05 = significant drift)
   - Pros: Non-parametric, works for any distribution
   - Cons: Sensitive to sample size

b) Chi-Square Test:
   - For: Categorical features
   - Tests: Whether frequency distributions differ
   - Output: p-value (< 0.05 = significant drift)
   - Pros: Standard statistical test
   - Cons: Requires sufficient samples per category

c) Population Stability Index (PSI):
   - For: Both numerical and categorical
   - Formula: Σ (actual% - expected%) × ln(actual% / expected%)
   - Thresholds: <0.1 (no drift), 0.1-0.2 (moderate), >0.2 (significant)
   - Pros: Industry standard, easy to interpret
   - Cons: Requires binning for numerical features

Simple Metrics:
---------------

d) Mean/Median Difference:
   - Compare: mean(new_data) vs mean(training_data)
   - Threshold: Typically 10-20% change
   - Pros: Easy to understand and explain
   - Cons: Doesn't capture distribution shape changes

e) Standard Deviation Change:
   - Compare: std(new_data) vs std(training_data)
   - Detects: Changes in variance
   - Pros: Catches spread changes
   - Cons: Sensitive to outliers


5. WHAT TO DO WHEN DRIFT IS DETECTED?
--------------------------------------

Immediate Actions:
1. Alert stakeholders
2. Investigate root cause
3. Assess business impact
4. Decide on intervention

Long-term Solutions:
1. Retrain model on recent data
2. Update feature engineering
3. Implement online learning
4. Adjust monitoring thresholds


6. INTERVIEW QUESTIONS & ANSWERS:
----------------------------------

Q: "How would you detect drift in a production ML system?"
A: "I'd implement automated drift detection using:
   1. Statistical tests (KS-test for numerical, Chi-square for categorical)
   2. Simple metrics (mean/std changes)
   3. Model performance monitoring (MAE, accuracy)
   4. Set up alerts when thresholds are exceeded
   5. Schedule regular drift reports"

Q: "What's the difference between data drift and concept drift?"
A: "Data drift is when input features (X) change distribution, but the 
   relationship X→Y stays the same. Concept drift is when the relationship 
   X→Y itself changes. Example: If student ages shift (data drift), the 
   model might still work. But if the grading system changes (concept drift), 
   the model's learned patterns become invalid."

Q: "How do you choose drift detection thresholds?"
A: "I'd:
   1. Start with statistical significance (p-value < 0.05)
   2. Analyze historical data to understand natural variation
   3. Consider business impact (false positives vs missed drift)
   4. Use domain knowledge (10% mean shift might be critical in finance, 
      but acceptable in marketing)
   5. Monitor and adjust based on false alert rate"

Q: "Can you have drift without performance degradation?"
A: "Yes! If the drift is in features that aren't important for predictions,
   or if the drift is within the training data's range, performance might
   not degrade. That's why we monitor both drift AND performance metrics."


7. BEST PRACTICES:
------------------

✓ Monitor continuously (daily/weekly)
✓ Use multiple detection methods
✓ Set up automated alerts
✓ Track drift over time (trend analysis)
✓ Document baseline statistics
✓ Version control drift thresholds
✓ Combine statistical tests with business metrics
✓ Have a retraining pipeline ready


8. COMMON PITFALLS:
-------------------

✗ Only monitoring model performance (reactive, not proactive)
✗ Using only one drift detection method
✗ Not considering seasonality (false positives)
✗ Thresholds too sensitive (alert fatigue)
✗ No automated response (manual intervention delays)
✗ Not tracking which features drift most


9. TOOLS & LIBRARIES:
---------------------

- Evidently AI: Drift detection and monitoring
- Alibi Detect: Advanced drift detection algorithms
- Great Expectations: Data quality + drift detection
- Custom: scipy.stats for KS-test, Chi-square
- MLflow: Model monitoring integration
- Prometheus + Grafana: Metrics visualization


10. REAL-WORLD EXAMPLE:
-----------------------

Student Performance Model:

Training Data (2023):
- Average age: 16.5
- Average absences: 5.2
- Pass rate: 75%

Production Data (2024):
- Average age: 17.8 (DRIFT DETECTED!)
- Average absences: 8.1 (DRIFT DETECTED!)
- Pass rate: 68% (PERFORMANCE DROP!)

Action:
1. Alert: "Significant drift in age and absences"
2. Investigate: New policy allowing older students
3. Impact: Model trained on younger students
4. Solution: Retrain on 2024 data
5. Result: Performance restored to 75% accuracy


SUMMARY FOR INTERVIEWS:
=======================

Key Points to Remember:
1. Data drift = input distribution changes
2. Detect using statistical tests (KS-test) + simple metrics (mean change)
3. Important because model performance degrades silently
4. Action: Alert → Investigate → Retrain
5. Best practice: Monitor continuously with automated alerts

One-Liner:
"Data drift is when production data looks different from training data,
detected through statistical tests, requiring model retraining to maintain
performance."
"""

# Now let's implement a production-ready drift detection module!
