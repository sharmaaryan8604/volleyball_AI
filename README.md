# Volleyball AI — Full Stack Deployment Guide

XGBoost + LightGBM + Markov chain hybrid for volleyball hit landing zone prediction.
**Stack**: FastAPI (Render) + React/Vite (Vercel)

---

## Project Structure

```
volleyball-ai/
├── api/
│   ├── app.py              ← FastAPI backend (wraps your src/ modules)
│   └── requirements.txt
├── src/                    ← Your existing ML modules (unchanged)
│   ├── preprocessing.py
│   ├── spatial_features.py
│   ├── ml_model.py
│   ├── markov_model.py
│   ├── simulation.py
│   └── evaluation.py
├── data/
│   ├── training data.csv
│   └── testing data.csv
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── PredictPage.jsx     ← Input form + court heatmap
│   │   │   ├── DashboardPage.jsx   ← Evaluation charts
│   │   │   └── SimulatePage.jsx    ← Markov scenario simulator
│   │   ├── components/
│   │   │   ├── Court.jsx           ← 25-zone SVG heatmap
│   │   │   └── Layout.jsx          ← Sidebar nav
│   │   ├── api.js                  ← API client
│   │   └── App.jsx
│   ├── .env.example
│   ├── package.json
│   └── vite.config.js
├── render.yaml             ← Render deployment config
├── vercel.json             ← Vercel deployment config
└── .gitignore
```

---

## Step 1 — Deploy Backend on Render

### 1.1 Push your full repo to GitHub
Make sure your repo contains **both** `api/` and `src/` and `data/`.

> **Important**: `data/` CSVs are excluded by `.gitignore` by default.
> Either remove the `data/*.csv` line from `.gitignore` OR use Git LFS for large files.
> Render needs the training data to build the model on cold start.

### 1.2 Create a new Web Service on Render
1. Go to [render.com](https://render.com) → **New → Web Service**
2. Connect your GitHub repo
3. Set these manually if `render.yaml` doesn't auto-fill:

| Field | Value |
|---|---|
| **Build Command** | `pip install -r api/requirements.txt` |
| **Start Command** | `uvicorn api.app:app --host 0.0.0.0 --port $PORT` |
| **Python Version** | 3.10 |

4. Click **Deploy**

### 1.3 Note your Render URL
It will look like: `https://volleyball-ai-api.onrender.com`

> **Cold start warning**: Render free tier spins down after inactivity.
> First request after sleep triggers model training (~60-90s).
> Consider upgrading to Starter ($7/mo) for persistent processes.

---

## Step 2 — Deploy Frontend on Vercel

### 2.1 Set the API URL environment variable
In your `frontend/` directory, copy `.env.example` to `.env.local`:
```bash
cp frontend/.env.example frontend/.env.local
# Edit and set:
VITE_API_URL=https://your-volleyball-ai-api.onrender.com
```

### 2.2 Deploy to Vercel
```bash
npm i -g vercel
vercel --cwd frontend
```
Or via the Vercel dashboard:
1. **New Project** → import your GitHub repo
2. Set **Root Directory** to `frontend`
3. Add Environment Variable: `VITE_API_URL` = your Render URL
4. Deploy

---

## Step 3 — Local Development

### Backend
```bash
cd volleyball-ai
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r api/requirements.txt
uvicorn api.app:app --reload --port 8000
```
API docs: http://localhost:8000/docs

### Frontend
```bash
cd frontend
npm install
# Create .env.local with VITE_API_URL=http://localhost:8000
npm run dev
```
App: http://localhost:5173

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Status check |
| POST | `/predict` | Predict landing zones (XGB+LGB+Markov blend) |
| POST | `/simulate` | Run Markov landing distribution simulation |
| GET | `/markov/info` | Number of learned Markov states |
| GET | `/zones/prior` | Zone prior distribution |
| GET | `/docs` | Auto-generated Swagger UI |

### Example `/predict` request
```json
{
  "hitter_location": 4,
  "set_location": 1,
  "pass_rating": 1,
  "num_blockers": 2,
  "block_touch": 0
}
```

### Example `/predict` response
```json
{
  "top1_zone": 12,
  "top3_zones": [12, 8, 15],
  "top5_zones": [12, 8, 15, 3, 21],
  "markov_hit": true,
  "all_probs": [0.02, 0.04, ...],
  "top_zones": [
    { "zone": 12, "probability": 0.187, "label": "Zone 12" },
    ...
  ]
}
```

---

## Troubleshooting

**`ModuleNotFoundError: src`**
→ Make sure Render's working directory is the repo root, not `api/`.
→ Add `PYTHONPATH=.` as an environment variable on Render.

**CORS errors in browser**
→ The backend already has `allow_origins=["*"]`. If still failing, check the Render URL matches what's in `VITE_API_URL`.

**Model takes too long on first request**
→ Normal on Render free tier. The model trains from scratch on cold start.
→ To fix: pre-train and serialize with `joblib.dump()`, load from file instead.

**Frontend build fails on Vercel**
→ Make sure `vercel.json` is in the repo root (not inside `frontend/`).
→ Root directory in Vercel project settings should be `frontend`.
