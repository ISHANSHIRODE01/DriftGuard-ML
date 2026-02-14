# 🚀 QUICK START - Dashboard V2.0

## ⚡ 3-Step Setup

### Step 1: Generate Test Data (30 seconds)
```bash
python dashboard/test_dashboard.py --severity moderate
```

### Step 2: Run Enhanced Dashboard (10 seconds)
```bash
streamlit run dashboard/dashboard_v2.py
```

### Step 3: Explore Features ✅
Open http://localhost:8501 and check:
- ✅ Drift score in sidebar
- ✅ Color-coded alerts
- ✅ Performance charts
- ✅ System logs

---

## 🧪 Test Different Scenarios

### Low Drift (Stable System)
```bash
python dashboard/test_dashboard.py --severity low
streamlit run dashboard/dashboard_v2.py
```
**Expected:** Green alerts, drift score < 20

### High Drift (Action Needed)
```bash
python dashboard/test_dashboard.py --severity high
streamlit run dashboard/dashboard_v2.py
```
**Expected:** Red alerts, drift score 50-75

### Critical Drift (Emergency)
```bash
python dashboard/test_dashboard.py --severity critical
streamlit run dashboard/dashboard_v2.py
```
**Expected:** Dark red alerts, drift score > 75

---

## 📊 What You'll See

### Sidebar
- Current model version
- Total retrainings
- Drift status with score
- Quick action buttons

### Main Dashboard
1. **System Alerts** - Real-time drift and performance warnings
2. **Performance Metrics** - 8 metric cards with improvements
3. **Visualizations** - MAE trend + Drift score trend charts
4. **System Logs** - 3 tabs with detailed history

---

## 🔄 Replace Original Dashboard

When ready for production:

```bash
# Backup original
cp dashboard/dashboard.py dashboard/dashboard_old.py

# Use v2 as main
cp dashboard/dashboard_v2.py dashboard/dashboard.py

# Deploy to Render
git add .
git commit -m "Upgrade to production dashboard v2.0"
git push origin main
```

---

## 🎯 Key Features

✅ **Drift Score**: Single 0-100 metric for drift severity  
✅ **Smart Alerts**: Color-coded warnings based on thresholds  
✅ **Historical Tracking**: Drift and performance trends over time  
✅ **Event Logging**: Complete audit trail of all system events  
✅ **Download Reports**: Export logs and drift details as CSV/JSON  
✅ **Live Predictions**: Test model with automatic logging  

---

## 📖 Full Documentation

See `docs/DASHBOARD_V2_GUIDE.md` for:
- Complete feature list
- Customization options
- Troubleshooting guide
- API reference

---

**You're ready to monitor your ML system like a pro!** 🎉
