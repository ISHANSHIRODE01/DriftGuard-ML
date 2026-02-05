# Evidently AI vs. Manual Drift Detection

**Date**: 2026-02-05  
**Topic**: Selecting the Right Drift Detection Strategy

---

## 1. When to Use Monitoring Tools (Evidently AI)

Evidently AI is a sophisticated, visual monitoring tool preferred when:

### ✅ You Need Visual Explainability
- **Why**: Stakeholders (PMs, business users) need to *see* the drift.
- **Evidently Feature**: Detailed interactive HTML reports with histograms, scatter plots, and distribution overlays.
- **Manual**: Just numbers (p-values, means). Harder to trust without visualization.

### ✅ Complex Statistical Tests
- **Why**: You need robust tests for different data types without writing boilerplate.
- **Evidently Feature**: Automatically selects tests (KS, Chi-square, Jensen-Shannon, Wasserstein Distance) based on feature type (numerical/categorical/text).
- **Manual**: Requires writing custom logic for each test and edge case (e.g., handling NaNs, zero variance).

### ✅ Production Dashboards
- **Why**: You want to track drift *over time* in a tool like Grafana or MLflow.
- **Evidently Feature**: JSON/Prometheus exports designed for time-series monitoring.
- **Manual**: Requires building custom logging infrastructure.

### ✅ Text & Unstructured Data
- **Why**: Numerical stats don't work on text.
- **Evidently Feature**: Comparison of embedding distributions, text length, sentiment drift.
- **Manual**: Extremely difficult to implement from scratch.

---

## 2. When to Use Manual Tests (Custom Scripts)

Your custom `DriftDetector` class is preferred when:

### ✅ Lightweight Dependencies
- **Why**: You are running in a constrained environment (e.g., Lambda function) where a heavy library like Evidently (with dependencies on Pandas, Plotly, etc.) is too much.
- **Evidently**: Heavy installation (~100MB+ dependencies).
- **Manual**: Lightweight (only requires Scipy/Numpy).

### ✅ Critical Blocking Gates
- **Why**: You need a binary "GO/NO-GO" decision in a pipeline.
- **Evidently**: Can do this, but setup is more complex for simple boolean decisions.
- **Manual**: Simple `if p_value < 0.05: raise Error` logic is clearer and faster (ms vs seconds).

### ✅ Specific Custom Thresholds
- **Why**: You have very specific business logic (e.g., "Drift is okay if mean shift < 5% UNLESS variance > 50%").
- **Evidently**: Configurable, but complex logic requires custom metrics.
- **Manual**: You have full control over the if/else logic.

---

## 3. Recommended Hybrid Approach

**Use Both:**
1.  **Pipeline Gate (Manual)**: Run your fast, lightweight `DriftDetector` in the CI/CD pipeline or API endpoint. Block deployment if critical features drift deeply.
2.  **Monitoring Dashboard (Evidently)**: Generate Evidently reports daily/weekly and push them to a dashboard for human review and deep debugging.

---

## Summary Comparison

| Feature | Evidently AI | Manual Tests |
| :--- | :--- | :--- |
| **Setup** | Easy (Presets) | Moderate (Code required) |
| **Visuals** | Excellent (Interactive) | None (Console/Logs) |
| **Speed** | Slower (seconds) | Fast (milliseconds) |
| **Deep Debugging**| Excellent | Limited |
| **Dependency Weight** | Heavy | Light |
| **Use Case** | Dashboarding, Exploratory | CI/CD Gates, Basic Checks |
