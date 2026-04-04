# Volleyball AI — Frontend

React + Vite frontend for the Volleyball AI prediction system.

## Quick Start

```bash
npm install
cp .env.example .env        # set VITE_API_URL
npm run dev
```

## Project Structure

```
src/
├── api/
│   └── client.js           # Axios client + payload cleaner (fixes 422)
├── components/
│   ├── Sidebar.jsx / .css
│   ├── Heatmap.jsx / .css  # Fixed heatmap with correct color interpolation
│   └── FormControls.jsx / .css
├── pages/
│   ├── Predict.jsx / .css  # Zone predictor + heatmap
│   ├── Simulate.jsx / .css # Rally simulator + bar chart
│   └── Dashboard.jsx / .css# Stats, radar, pie charts
├── utils/
│   └── constants.js        # Zone layout, color helpers
├── App.jsx / .css
├── main.jsx
└── index.css               # Design tokens (CSS vars)
```

## Fix for 422 Predict Failed

The `cleanPayload()` in `src/api/client.js` strips out:
- `null` / `undefined` values
- Empty strings `""`
- The sentinel value `"any"` (from optional dropdowns)
- Converts `num_blockers` from string → integer

This prevents FastAPI/Pydantic from receiving invalid enum members.

## Environment Variables

| Variable        | Default                                  | Description           |
|-----------------|------------------------------------------|-----------------------|
| `VITE_API_URL`  | `https://volleyball-ai.onrender.com`     | Backend base URL      |

## Build for Production

```bash
npm run build     # outputs to dist/
npm run preview   # preview the build
```
