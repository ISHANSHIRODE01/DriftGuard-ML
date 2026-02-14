# 📸 Screenshot Capture Guide

## Required Screenshots for README

To complete your GitHub README, you need to capture these 6 screenshots:

---

## 1. Dashboard Overview
**File**: `docs/screenshots/dashboard_overview.png`

**What to capture:**
- Full dashboard view showing:
  - Sidebar with system status
  - Alert banners at top
  - Performance metrics (8 cards)
  - Both trend charts (MAE + Drift)

**How to capture:**
1. Run: `streamlit run dashboard/dashboard_v2.py`
2. Wait for page to fully load
3. Scroll to show all sections
4. Take full-page screenshot
5. Save as `dashboard_overview.png`

---

## 2. Alert System
**File**: `docs/screenshots/alerts.png`

**What to capture:**
- Close-up of the alert section showing:
  - Drift alert card (red/orange/green)
  - Performance alert card
  - Severity indicators
  - Drift score

**How to capture:**
1. Generate high drift: `python dashboard/test_dashboard.py --severity high`
2. Refresh dashboard
3. Zoom in on "System Alerts" section
4. Capture just the alert cards
5. Save as `alerts.png`

---

## 3. Performance Metrics
**File**: `docs/screenshots/metrics.png`

**What to capture:**
- The 8 metric cards showing:
  - Current MAE
  - Current R² Score
  - Previous MAE
  - Improvement %
  - Drift Score
  - Features Analyzed
  - Drifted Features
  - Drift Percentage

**How to capture:**
1. Ensure metadata has multiple versions
2. Focus on the metrics section
3. Capture all 8 cards in one shot
4. Save as `metrics.png`

---

## 4. Drift Analysis
**File**: `docs/screenshots/drift_analysis.png`

**What to capture:**
- The drift details tab showing:
  - Feature-level drift table
  - Statistical test results
  - P-values
  - Drift reasons

**How to capture:**
1. Go to "System Logs & History" tab
2. Click "Drift Details" sub-tab
3. Ensure "Show all features" is unchecked (shows only drifted)
4. Capture the table
5. Save as `drift_analysis.png`

---

## 5. System Logs
**File**: `docs/screenshots/system_logs.png`

**What to capture:**
- The logs panel showing:
  - Recent events with timestamps
  - Different severity levels (colored dots)
  - Event types
  - Retraining history table

**How to capture:**
1. Go to "System Logs & History" tab
2. Click "Recent Events" sub-tab
3. Capture the log entries
4. Alternatively, capture "Retraining History" table
5. Save as `system_logs.png`

---

## 6. Live Predictions
**File**: `docs/screenshots/predictions.png`

**What to capture:**
- The prediction form showing:
  - Input fields filled with sample data
  - Prediction result displayed
  - Success message or performance indicator

**How to capture:**
1. Scroll to "Live Prediction & Testing"
2. Expand the prediction form
3. Fill in sample data
4. Click "Predict Grade"
5. Capture the result
6. Save as `predictions.png`

---

## 📐 Screenshot Specifications

### Recommended Settings:
- **Resolution**: 1920x1080 or higher
- **Format**: PNG (for transparency and quality)
- **File Size**: < 500KB each (compress if needed)
- **Browser**: Chrome or Firefox (for consistent rendering)

### Tools:
- **Windows**: Snipping Tool, Snip & Sketch, or ShareX
- **Mac**: Cmd+Shift+4 (area selection)
- **Browser Extension**: Awesome Screenshot, Nimbus

---

## 🎨 Tips for Great Screenshots

1. **Clean Browser**: Remove bookmarks bar, extensions
2. **Full Screen**: Use F11 for distraction-free capture
3. **Zoom Level**: 100% (Ctrl+0 to reset)
4. **Dark Mode**: Streamlit dark theme looks professional
5. **Data Variety**: Show interesting drift patterns
6. **Annotations**: Add arrows or highlights if needed (optional)

---

## 🚀 Quick Capture Workflow

```bash
# Step 1: Generate test data with high drift
python dashboard/test_dashboard.py --severity high

# Step 2: Start dashboard
streamlit run dashboard/dashboard_v2.py

# Step 3: Open browser at http://localhost:8501

# Step 4: Capture screenshots in this order:
# 1. Full dashboard (overview)
# 2. Alert section (zoom in)
# 3. Metrics section
# 4. Drift Details tab
# 5. System Logs tab
# 6. Make a prediction and capture result

# Step 5: Save all to docs/screenshots/
```

---

## 📁 Directory Structure

After capturing, your structure should be:

```
docs/
└── screenshots/
    ├── dashboard_overview.png
    ├── alerts.png
    ├── metrics.png
    ├── drift_analysis.png
    ├── system_logs.png
    └── predictions.png
```

---

## ✅ Verification Checklist

Before committing:
- [ ] All 6 screenshots captured
- [ ] Files are PNG format
- [ ] Each file < 500KB
- [ ] Images are clear and readable
- [ ] No sensitive data visible
- [ ] Saved in `docs/screenshots/` directory
- [ ] README.md references updated (if needed)

---

## 🔄 Alternative: Use Placeholder Images

If you can't capture screenshots immediately, you can use placeholder images temporarily:

1. Create simple placeholder images with text
2. Replace them later with actual screenshots
3. Or use tools like [placeholder.com](https://placeholder.com/)

Example placeholder:
```
https://via.placeholder.com/1200x600/667eea/ffffff?text=Dashboard+Overview
```

---

## 📤 After Capturing

1. Optimize images (use TinyPNG or similar)
2. Commit to Git:
```bash
git add docs/screenshots/
git commit -m "Add dashboard screenshots"
git push origin main
```

3. Verify images display correctly on GitHub
4. Update README if paths need adjustment

---

**Your README will look professional and complete with these screenshots!** 📸
