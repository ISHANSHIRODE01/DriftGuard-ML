# 🚀 RENDER DEPLOYMENT - QUICK START

## ⚡ 3-Step Deployment

### 1️⃣ Push to GitHub
```bash
git add .
git commit -m "Ready for Render deployment"
git push origin main
```

### 2️⃣ Deploy on Render
1. Go to: https://dashboard.render.com
2. Click: **New +** → **Web Service**
3. Connect your GitHub repo
4. Use these settings:

**Build Command:**
```bash
pip install -r requirements.txt
```

**Start Command:**
```bash
streamlit run dashboard/dashboard.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
```

**Environment Variables:**
- `MONGO_URI` = Your MongoDB connection string
- `PYTHON_VERSION` = 3.9.0

### 3️⃣ Click Deploy!

Your app will be live at: `https://your-app-name.onrender.com`

---

## 🔑 MongoDB Setup (If needed)

1. Go to: https://www.mongodb.com/cloud/atlas
2. Create FREE cluster
3. Create database user
4. **Network Access:** Add `0.0.0.0/0`
5. Copy connection string
6. Paste in Render environment variables

---

## ✅ Verification

Once deployed, your dashboard should show:
- ✅ Model version in sidebar
- ✅ Green "STABLE" status
- ✅ Working prediction form

---

**That's it! You're live! 🎉**

For detailed troubleshooting, see `DEPLOYMENT_GUIDE.md`
