# 🥊 Boxing Biomechanics Lab

Interactive web app for kinematic and kinetic analysis of boxing strikes using **Noraxon** IMU + force sensor data.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

---

## Features / Características

- 📁 **Upload** Noraxon Excel files directly in the browser
- 🎯 **Auto-detect** up to 10 strike events by peak force
- 📊 **Interactive Plotly charts** — zoom, hover, pan
- 🌐 **Bilingual** — Español / English
- 📈 **Metrics computed per strike:**
  - Peak Force, Net Peak Force (N)
  - Time to Peak, Contact Duration (ms)
  - Total Impulse, Impulse to Peak (N·s)
  - Max RFD, Max Jerk (N/s, N/s²)
  - Force @ 10ms, Force @ 20ms (N)
  - RFD 0–10ms, RFD 0–20ms (N/s)
  - Impact Velocity, Max Velocity (m/s)
  - Acceleration at Impact (m/s²)
- 💾 **Export** all results to Excel (summary + per-event raw data)

---

## Data Format

The app expects an Excel file exported from Noraxon with the following columns:

| Column | Description | Unit |
|--------|-------------|------|
| `Time` | Timestamp | s |
| `1x`, `1y`, `1z` | Accelerometer 1 (fist) | milli-g |
| `2x`, `2y`, `2z` | Accelerometer 2 | milli-g |
| `3x`, `3y`, `3z` | Accelerometer 3 | milli-g |
| `fx`, `fy`, `fz` | Force sensor | N |

---

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Deploy to Streamlit Cloud (Free)

1. **Fork or push** this repo to your GitHub account
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **"New app"**
4. Select your repo → branch `main` → file `app.py`
5. Click **"Deploy"** — your URL will be ready in ~2 minutes

---

## Tech Stack

- [Streamlit](https://streamlit.io) — web framework
- [Plotly](https://plotly.com/python/) — interactive charts
- [Pandas](https://pandas.pydata.org) + [NumPy](https://numpy.org) — data analysis
- [OpenPyXL](https://openpyxl.readthedocs.io) — Excel export

---

## License

MIT
