# 🏐 Volleyball AI — Attack Landing Zone Prediction

> **XGBoost + LightGBM + Markov Chain Hybrid**  
> 58.08% Top-1 Accuracy | 0.701 MRR | 0.0183 ECE | +49.73pp lift over majority baseline

A full-stack machine learning system that predicts **where a volleyball attack will land** (out of 25 court zones) given rally context — hitter position, pass quality, set type, and blocker pressure. Built for tactical coaching applications.

🔗 **Live Demo**: [volleyball-ai-black.vercel.app](https://volleyball-ai-black.vercel.app)  
🔗 **Backend API**: [volleyball-ai.onrender.com](https://volleyball-ai.onrender.com)  
📖 **API Docs**: [volleyball-ai.onrender.com/docs](https://volleyball-ai.onrender.com/docs)

---

## 📊 Results at a Glance

| Metric | Value |
|---|---|
| Top-1 Accuracy | **58.08%** |
| Top-2 Accuracy | 71.93% |
| Top-3 Accuracy | **77.88%** |
| Top-5 Accuracy | 84.52% |
| Top-10 Accuracy | 92.87% |
| Mean Reciprocal Rank (MRR) | **0.701** |
| Expected Calibration Error (ECE) | **0.0183** |
| Log Loss | 1.7259 |
| Lift over majority baseline | **+49.73pp** |
| Markov hit rate | 100% (3,289/3,289) |
| Training samples | 14,660 |
| Validation samples | 3,702 |
| Features used | 32 |
| Markov states learned | 143 |

The system is **27× better than random** (2.11% baseline → 58.08%) and correctly places the true landing zone in the top-3 predictions for nearly **4 out of 5 attacks**.

---

## 🗂️ Project Structure

```
volleyball-ai/
├── api/
│   ├── app.py              ← FastAPI backend (wraps src/ ML modules)
│   └── requirements.txt
├── src/                    ← Core ML pipeline
│   ├── preprocessing.py    ← Zone binning, categorical encoding, cleaning
│   ├── spatial_features.py ← 18 engineered spatial/geometric features
│   ├── ml_model.py         ← XGBoost + LightGBM sklearn Pipelines
│   ├── markov_model.py     ← Markov chain transition matrix + fallback
│   ├── hybrid_model.py     ← Probability blending (ML + Markov + prior)
│   ├── simulation.py       ← Monte Carlo simulations (5 scenario types)
│   ├── evaluation.py       ← Per-zone accuracy, confusion, MRR breakdown
│   └── attack_map.py       ← Probability-weighted arrow visualisation
├── data/
│   ├── training_data.csv
│   └── testing_data.csv
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── PredictPage.jsx     ← Input form + court heatmap
│   │   │   ├── DashboardPage.jsx   ← Evaluation charts
│   │   │   └── SimulatePage.jsx    ← Markov scenario simulator
│   │   ├── components/
│   │   │   ├── Court.jsx           ← 25-zone SVG heatmap
│   │   │   └── Layout.jsx          ← Sidebar navigation
│   │   ├── api.js                  ← API client
│   │   └── App.jsx
│   ├── .env.example
│   ├── package.json
│   └── vite.config.js
├── main.py                 ← Training entry point
├── pre_train.py            ← Pre-serialise models with joblib
├── render.yaml             ← Render deployment config
├── requirements.txt
└── runtime.txt
```

---

## 🧠 How It Works

### Problem

Given a volleyball rally context, predict which of **25 court zones** the attack will land in — a 25-class classification problem. Random guessing gives 4%; the majority-class baseline gives 8.35%.

### Three-Component Hybrid Architecture

```
Input Features (32)
       │
       ├──► XGBoost (multi:softprob, 500 trees)  ┐
       │                                          ├── 0.5/0.5 average ──► ML probs
       └──► LightGBM (leaf-wise, 500 trees)       ┘
                                                        │
                                                        ▼
                                              70% ML + 30% Markov
                                                        │
Markov Chain ──► P(zone | pass, set, hitter) ──────────┘
                                                        │
                                                        ▼
                                              90% blend + 10% zone prior
                                                        │
                                                        ▼
                                           Final probability distribution
                                           over 25 landing zones
```

**Why blend?** XGBoost/LightGBM learn complex non-linear interactions across 32 features; the Markov chain preserves exact empirical conditional frequencies from historical data. Neither alone is as strong as the combination.

---

## ⚙️ Core Components

### 1. Data Preprocessing (`preprocessing.py`)

- Bins floating-point DataVolley zone coordinates to integer zone IDs (1–15 attack, 1–25 landing)
- Uses pandas `Int64` (nullable integer) to preserve `NaN` for downstream imputers
- Encodes categoricals as integers for Markov dictionary key compatibility:
  - `pass_rating`: `"in"→1`, `"out"→0`
  - `set_location`: `"outside"→1`, `"oppo"→2`, `"quick"→3`, `"bic"→4`, `"dump"→5`, ...
  - `block_touch`: `"yes"→1`, `"no"→0`

### 2. Spatial Feature Engineering (`spatial_features.py`)

18 engineered features transform raw zone IDs into geometric signals:

| Feature | Formula | Why Useful |
|---|---|---|
| `hx`, `hy` | `(zone-1)%5`, `(zone-1)//5` | Hitter grid position (col, row) |
| `dist_net` | `= hy` | Rows from net; back-row attackers have wider angles |
| `dist_center` | `√((hx-2)²+(hy-1)²)` | Corner vs. middle attacker |
| `attack_dx/dy` | `hitter - setter` coords | Direction of the set (pull vs. push) |
| `attack_angle` | `arctan2(dy, dx)` | Set angle predicts attack direction |
| `cross_court` | `hx >= 3 → 1` | Binary: right-side hitter |
| `line_attack` | `hx <= 1 → 1` | Binary: left-side hitter |
| `back_row` | `hy >= 2 → 1` | Back-row vs. front-row profile |
| `attack_pressure` | `num_blockers²` | Quadratic blocker pressure (2 blockers = 4× pressure) |
| `quality_pressure` | `pass_rating / (1 + blockers)` | Pass quality under defensive pressure |

> Setter coordinates use a lookup dictionary (`SET_LOC_COORDS`), not grid arithmetic — because `set_location` codes are categorical, not sequential zone IDs.

### 3. ML Models (`ml_model.py`)

Both models are wrapped in `sklearn.Pipeline` with a `ColumnTransformer` that applies median imputation (numerics) and one-hot encoding (categoricals), fit on training data only to prevent leakage.

**XGBoost**: sequential tree ensemble, `multi:softprob` objective, `mlogloss` evaluation  
**LightGBM**: leaf-wise growth, GOSS sampling, EFB feature bundling — faster on large datasets

Key shared hyperparameters: `n_estimators=500`, `learning_rate=0.05`, `subsample=0.9`, `colsample_bytree=0.9`

### 4. Markov Chain (`markov_model.py`)

A conditional frequency table: `P(landing_zone | pass_rating, set_location, hitter_location)`.

- State space: 2 × 8 × 15 = 240 possible states; **143 states learned** (those with ≥5 examples)
- **Three-level fallback chain** for unseen states:
  1. Full: `(pass_rating, set_location, hitter_location)`
  2. Partial: `(None, set_location, hitter_location)` — drop pass quality
  3. Minimal: `(None, None, hitter_location)` — hitter zone only
- Achieves **100% hit rate** on validation (3,289/3,289)

### 5. Hybrid Blending (`hybrid_model.py`)

```python
# Step 1: ML ensemble
probs = 0.5 * xgb_probs + 0.5 * lgb_probs

# Step 2: Markov blend (per sample)
probs[zone] = 0.70 * probs[zone] + 0.30 * markov_probs[zone]

# Step 3: Zone prior smoothing (Laplace)
probs[zone] = 0.90 * probs[zone] + 0.10 * zone_prior[zone]
```

### 6. Train/Validation Split (`main.py`)

Split is performed on **rally IDs**, not rows. Each rally (a sequence of serve → receive → pass → set → hit) is entirely in training or validation — never split across the boundary. This prevents data leakage from consecutive rows of the same rally appearing in both sets.

---

## 🔬 Evaluation Insights

### Per-Zone Difficulty

Hardest zones to predict and why:

| Zone | Top-1 Acc | Reason |
|---|---|---|
| Zone 19 | 28.21% | Very few samples |
| Zone 23 | 38.64% | Out-of-bounds adjacent; true uncertainty |
| Zone 9 | 48.91% | Middle attacker spreads to 10+ landing zones |

### Confusion Analysis

All top-10 most common errors are **spatially adjacent zone confusions** (e.g. Zone 2 predicted as Zone 3). The model never predicts a far-corner zone when the true answer is the opposite corner — confirming that spatial features correctly encode court geometry.

### Key Insight from Simulations

Pass quality has a dramatic effect on predictability. For a Zone 11 hitter with outside set:
- **Good pass**: 58.63% probability concentrated in Zone 3
- **Bad pass**: distribution spreads across 8+ zones, Zone 3 drops to 2.25%

---

## 🚀 Deployment

### Backend — FastAPI on Render

1. Push the full repo to GitHub (include `data/` CSVs or use Git LFS)
2. Create a new **Web Service** on [render.com](https://render.com)
3. Connect your GitHub repo and set:

| Field | Value |
|---|---|
| Build Command | `pip install -r requirements.txt` |
| Start Command | `uvicorn api.app:app --host 0.0.0.0 --port $PORT` |
| Python Version | 3.10 |
| Environment Variable | `PYTHONPATH=.` |

> **Cold start note**: Render free tier spins down after inactivity. First request triggers model training (~60–90s). Consider pre-training and serialising with `joblib` (see `pre_train.py`) or upgrading to Render Starter ($7/mo).

### Frontend — React/Vite on Vercel

```bash
cd frontend
cp .env.example .env.local
# Set VITE_API_URL=https://your-render-url.onrender.com

npm i -g vercel
vercel
```

Or via the Vercel dashboard: import the repo, set `VITE_API_URL` as an environment variable, deploy.

---

## 💻 Local Development

### Backend

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn api.app:app --reload --port 8000
# Docs: http://localhost:8000/docs
# Production API: https://volleyball-ai.onrender.com/docs
```

### Frontend

```bash
cd frontend
npm install
# Create frontend/.env.local:
# VITE_API_URL=http://localhost:8000
npm run dev
# App: http://localhost:5173
```

### Run Training Pipeline

```bash
python main.py
```

---

## 🔌 API Reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Status check |
| `POST` | `/predict` | Predict landing zones (hybrid blend) |
| `POST` | `/simulate` | Monte Carlo landing distribution |
| `GET` | `/markov/info` | Number of learned Markov states |
| `GET` | `/zones/prior` | Zone prior distribution |
| `GET` | `/docs` | Swagger UI |

### `/predict` — Request

```json
{
  "hitter_location": 11,
  "set_location": 1,
  "pass_rating": 1,
  "num_blockers": 1,
  "block_touch": 0
}
```

### `/predict` — Response

```json
{
  "top1_zone": 3,
  "top3_zones": [3, 15, 16],
  "top5_zones": [3, 15, 16, 14, 8],
  "markov_hit": true,
  "all_probs": [0.02, 0.04, "..."],
  "top_zones": [
    { "zone": 3, "probability": 0.587, "label": "Zone 3" },
    { "zone": 15, "probability": 0.170, "label": "Zone 15" }
  ]
}
```

---

## 🛠️ Troubleshooting

**`ModuleNotFoundError: src`**  
→ Add `PYTHONPATH=.` as an environment variable on Render.

**CORS errors in browser**  
→ Backend uses `allow_origins=["*"]`. Verify the Render URL matches `VITE_API_URL` exactly (no trailing slash).

**Model takes too long on first request**  
→ Expected on Render free tier (cold start). Run `pre_train.py` to serialise models with `joblib` so they load from disk instead of training from scratch.

**Frontend build fails on Vercel**  
→ Ensure `vercel.json` is in the repo root. Set the Vercel project root to the repo root (not `frontend/`).

---

## 📦 Tech Stack

| Layer | Technology |
|---|---|
| ML Models | XGBoost, LightGBM, scikit-learn |
| Data | pandas, numpy |
| Spatial Features | Custom geometry (arctan2, Euclidean distance) |
| Simulation | NumPy Monte Carlo (n=10,000) |
| Backend | FastAPI, Uvicorn |
| Frontend | React, Vite |
| Backend Hosting | Render |
| Frontend Hosting | Vercel |

---

## 📄 License

MIT

---

*For a full technical breakdown of every component — math, code, hyperparameter rationale, and engineering fixes — see the [Full Technical Explanation PDF](./volleyball_ai_explained.pdf).*
