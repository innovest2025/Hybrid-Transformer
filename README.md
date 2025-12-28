# Hybrid Transformer–Autoencoder Crop Stress Prototype (Python + Firebase)

This is a **Python-only** hackathon prototype:
- **Streamlit (Python UI)**
- **PyTorch Hybrid Transformer–Autoencoder** for anomaly detection
- **Firebase**: Firestore storage + optional Email/Password Auth via Firebase REST API

## 1) Setup

### Create a virtualenv (recommended)
```bash
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### Install deps
```bash
pip install -r requirements.txt
```

## 2) Firebase Configuration (Required for saving runs/history)

### Firestore (server-side)
1. In Firebase Console: Project Settings → Service Accounts
2. Generate a **service account JSON**
3. Put it somewhere safe and set an env var:

**Windows PowerShell**
```powershell
$env:FIREBASE_SERVICE_ACCOUNT_JSON="C:\path\to\serviceAccount.json"
```

**macOS/Linux**
```bash
export FIREBASE_SERVICE_ACCOUNT_JSON="/path/to/serviceAccount.json"
```

### Optional: Email/Password Login (client-side)
If you want real login, set:
- `FIREBASE_API_KEY` (Web API key from Firebase console)
- Enable Email/Password provider: Authentication → Sign-in method

**Windows PowerShell**
```powershell
$env:FIREBASE_API_KEY="YOUR_WEB_API_KEY"
```

**macOS/Linux**
```bash
export FIREBASE_API_KEY="YOUR_WEB_API_KEY"
```

> If you don't set `FIREBASE_API_KEY`, the app will run in **Demo mode** with a local session user.

## 3) Run the App
```bash
streamlit run app.py
```

## 4) Data Format
Upload a CSV with:
- `timestamp` (any parseable date/time OR an integer index)
- `value` (e.g., NDVI / soil moisture / temperature / vegetation index)

Example:
```csv
timestamp,value
2025-01-01,0.72
2025-01-02,0.71
...
```

## 5) How the MVP model works
- Trains quickly on the first part of the uploaded sequence as "normal"
- Uses **reconstruction error** as anomaly score
- Chooses threshold from training errors (95th percentile by default)
- Flags anomalies + produces risk label (Low/Medium/High)
