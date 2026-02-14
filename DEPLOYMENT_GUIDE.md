# 🚀 RENDER DEPLOYMENT GUIDE - DriftGuard-ML

## 📋 Pre-Deployment Checklist

### ✅ Files Ready
- [x] `requirements.txt` - Updated with dnspython
- [x] `render.yaml` - Infrastructure as Code config
- [x] `runtime.txt` - Python 3.9.0 specified
- [x] Model files in `model/` directory
- [x] Dashboard entry point: `dashboard/dashboard.py`

---

## 🌐 STEP-BY-STEP DEPLOYMENT

### Step 1: Push to GitHub

```bash
# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit
git commit -m "Deploy DriftGuard-ML to Render"

# Create main branch
git branch -M main

# Add remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/DriftGuard-ML.git

# Push
git push -u origin main
```

---

### Step 2: Setup MongoDB Atlas (If not done)

1. Go to [mongodb.com/cloud/atlas](https://www.mongodb.com/cloud/atlas)
2. Create a **FREE** cluster
3. Create a database user with password
4. **CRITICAL:** In Network Access, add IP: `0.0.0.0/0` (Allow from anywhere)
5. Get your connection string (looks like):
   ```
   mongodb+srv://username:password@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority
   ```

---

### Step 3: Deploy on Render

#### Option A: Using render.yaml (Recommended)

1. Go to [dashboard.render.com](https://dashboard.render.com)
2. Click **New +** → **Blueprint**
3. Connect your GitHub repository
4. Render will auto-detect `render.yaml`
5. **Add Environment Variable:**
   - Key: `MONGO_URI`
   - Value: `your-mongodb-connection-string`
6. Click **Apply**

#### Option B: Manual Setup

1. Go to [dashboard.render.com](https://dashboard.render.com)
2. Click **New +** → **Web Service**
3. Connect your GitHub repository
4. **Configure:**
   - **Name:** `driftguard-dashboard`
   - **Environment:** `Python 3`
   - **Region:** `Oregon (US West)` or closest to you
   - **Branch:** `main`
   - **Build Command:**
     ```bash
     pip install -r requirements.txt
     ```
   - **Start Command:**
     ```bash
     streamlit run dashboard/dashboard.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
     ```
5. **Environment Variables:**
   - Click **Advanced**
   - Add: `MONGO_URI` = `your-connection-string`
   - Add: `PYTHON_VERSION` = `3.9.0`
6. Click **Create Web Service**

---

### Step 4: Monitor Deployment

1. Render will show build logs in real-time
2. Build takes ~3-5 minutes
3. Look for: `✓ Build successful`
4. Then: `Your service is live at https://driftguard-dashboard.onrender.com`

---

## 🔧 TROUBLESHOOTING

### Issue: Build fails with "No module named 'dnspython'"
**Fix:** Already added to requirements.txt ✅

### Issue: Streamlit not binding to port
**Fix:** Start command includes `--server.port $PORT` ✅

### Issue: MongoDB connection timeout
**Fix:** Check MongoDB Atlas Network Access allows `0.0.0.0/0`

### Issue: Model files not found
**Fix:** Ensure `model/model.pkl` and `model/preprocessor.pkl` are committed to Git

---

## 🎯 POST-DEPLOYMENT

### Test Your Live App
1. Visit: `https://your-app-name.onrender.com`
2. Check sidebar shows: **🟢 STABLE** status
3. Go to **Live Prediction & Ingestion** tab
4. Submit a test prediction
5. Verify it returns a G3 score

### Monitor Performance
- Render Dashboard → Metrics
- Check response times
- Monitor memory usage

---

## 📊 EXPECTED BEHAVIOR

### First Launch (Cold Start)
- Takes ~30-60 seconds
- Dashboard auto-bootstraps if model missing
- Shows "First-time setup: Training baseline model..."

### Normal Operation
- Dashboard loads in ~5-10 seconds
- Predictions return in <1 second
- Drift reports update when pipeline runs

---

## 🚨 IMPORTANT NOTES

1. **Free Tier Limits:**
   - App sleeps after 15 min of inactivity
   - First request after sleep takes ~30s (cold start)
   - 750 hours/month free

2. **Database:**
   - MongoDB Atlas Free Tier: 512MB storage
   - Sufficient for ~100K predictions

3. **Model Files:**
   - Current model size: ~1MB
   - Well within Render's limits

---

## 🎉 SUCCESS CRITERIA

✅ Build completes without errors
✅ App is accessible at Render URL
✅ Dashboard shows model version in sidebar
✅ Prediction form works and returns results
✅ MongoDB logs predictions (check Atlas)

---

**You're ready to deploy! 🚀**

If you encounter any issues, check the build logs in Render Dashboard.
