import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Boxing Biomechanics Lab",
    page_icon="🥊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Hero header */
.hero {
    background: linear-gradient(135deg, #1A1A2E 0%, #C62828 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    color: white;
}
.hero h1 { font-size: 2.2rem; font-weight: 800; margin: 0; letter-spacing: -0.5px; }
.hero p  { font-size: 1rem; margin: 0.4rem 0 0; opacity: 0.85; }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1A1A2E 0%, #16213E 100%);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    color: white;
    text-align: center;
    border-left: 4px solid #C62828;
    margin-bottom: 0.5rem;
}
.metric-card .label { font-size: 0.72rem; font-weight: 600; opacity: 0.7; text-transform: uppercase; letter-spacing: 0.5px; }
.metric-card .value { font-size: 1.9rem; font-weight: 800; margin: 0.2rem 0; color: #F9A825; }
.metric-card .unit  { font-size: 0.8rem; opacity: 0.7; }

/* Gold card variant */
.metric-card-gold {
    background: linear-gradient(135deg, #7B3F00 0%, #F9A825 100%);
    border-left: 4px solid #F9A825;
}
.metric-card-gold .value { color: white; }

/* Section title */
.section-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #1A1A2E;
    border-left: 4px solid #C62828;
    padding-left: 0.7rem;
    margin: 1.2rem 0 0.8rem;
}

/* Upload area style */
.stFileUploader { border: 2px dashed #C62828 !important; border-radius: 10px; }

/* Tab styling */
button[data-baseweb="tab"] { font-weight: 600 !important; font-size: 0.9rem !important; }
button[data-baseweb="tab"][aria-selected="true"] { color: #C62828 !important; border-bottom-color: #C62828 !important; }

/* Sidebar */
section[data-testid="stSidebar"] { background: #1A1A2E !important; }
section[data-testid="stSidebar"] * { color: white !important; }
section[data-testid="stSidebar"] .stSlider > div > div { background: #C62828 !important; }
section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.15) !important; }

/* Event badge */
.event-badge {
    display: inline-block;
    background: #C62828;
    color: white;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 700;
    margin-right: 0.5rem;
}

/* Summary table */
.stDataFrame { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# LANGUAGE STRINGS
# ─────────────────────────────────────────────────────────────────────────────
LANG = {
    "es": {
        "title": "Boxing Biomechanics Lab",
        "subtitle": "Análisis cinemático y cinético de golpes — Datos Noraxon",
        "lang_label": "Idioma / Language",
        "upload_label": "Subir archivo Excel (.xlsx)",
        "upload_help": "Archivo exportado desde Noraxon con columnas: Time, 1x/1y/1z, fx/fy/fz",
        "athlete_label": "Nombre del atleta",
        "athlete_placeholder": "Ej: K001_J",
        "params_title": "Parámetros de análisis",
        "num_events": "Número de golpes a detectar",
        "force_thresh": "Umbral mínimo de fuerza (N)",
        "min_sep": "Separación mínima entre golpes (s)",
        "window": "Ventana de análisis (s)",
        "acc_thresh": "Umbral de inicio — aceleración (m/s²)",
        "onset_thresh": "Umbral de contacto — fuerza (N)",
        "analyze_btn": "🥊 Analizar",
        "tab_signal": "📊 Señal Completa",
        "tab_events": "🎯 Por Evento",
        "tab_compare": "📈 Comparación",
        "tab_export": "💾 Exportar",
        "detected": "golpes detectados",
        "avg_force": "Fuerza media pico",
        "avg_vel": "Velocidad media impacto",
        "avg_rfd": "RFD media máx",
        "select_event": "Seleccionar golpe",
        "event_label": "Golpe",
        "peak_force": "Fuerza Pico",
        "net_force": "Fuerza Neta Pico",
        "time_to_peak": "Tiempo hasta Pico",
        "contact_dur": "Duración Contacto",
        "impulse_total": "Impulso Total",
        "impulse_peak": "Impulso → Pico",
        "rfd_max": "RFD Máx",
        "jerk_max": "Jerk Máx",
        "f_10ms": "Fuerza @ 10ms",
        "rfd_10": "RFD 0–10ms",
        "f_20ms": "Fuerza @ 20ms",
        "rfd_20": "RFD 0–20ms",
        "vel_impact": "Velocidad en Impacto",
        "vel_max": "Velocidad Máxima",
        "acc_impact": "Aceleración en Impacto",
        "strike_dur": "Duración Golpe (cin.)",
        "full_metrics": "Ver todas las métricas",
        "fig_force": "Fuerza Resultante",
        "fig_acc": "Aceleración Resultante",
        "fig_vel": "Velocidad Resultante del Puño",
        "fig_vel_xyz": "Velocidad por Ejes",
        "fig_rfd": "Rate of Force Development (RFD)",
        "onset": "Inicio contacto",
        "peak": "Pico fuerza",
        "end": "Fin contacto",
        "acc_onset": "Inicio golpe",
        "compare_title": "Comparación entre Golpes",
        "summary_table": "Tabla Resumen",
        "export_title": "Descargar Resultados",
        "download_excel": "⬇️ Descargar Excel de resultados",
        "download_raw": "⬇️ Datos del golpe",
        "upload_prompt": "👆 Sube un archivo Excel en la barra lateral para comenzar",
        "upload_prompt_sub": "Formato Noraxon: columnas Time, 1x/1y/1z, fx/fy/fz",
        "time_axis": "Tiempo (ms)",
        "full_time": "Tiempo (s)",
        "vel_impact_card": "Vel. en Impacto",
        "comparison_note": "Valores a t = tiempo del pico de fuerza",
    },
    "en": {
        "title": "Boxing Biomechanics Lab",
        "subtitle": "Kinematic & kinetic strike analysis — Noraxon Data",
        "lang_label": "Idioma / Language",
        "upload_label": "Upload Excel file (.xlsx)",
        "upload_help": "Noraxon export with columns: Time, 1x/1y/1z, fx/fy/fz",
        "athlete_label": "Athlete name",
        "athlete_placeholder": "e.g. K001_J",
        "params_title": "Analysis parameters",
        "num_events": "Number of strikes to detect",
        "force_thresh": "Minimum force threshold (N)",
        "min_sep": "Minimum separation between strikes (s)",
        "window": "Analysis window (s)",
        "acc_thresh": "Strike onset threshold — acceleration (m/s²)",
        "onset_thresh": "Contact onset threshold — force (N)",
        "analyze_btn": "🥊 Analyze",
        "tab_signal": "📊 Full Signal",
        "tab_events": "🎯 Per Event",
        "tab_compare": "📈 Comparison",
        "tab_export": "💾 Export",
        "detected": "strikes detected",
        "avg_force": "Mean peak force",
        "avg_vel": "Mean impact velocity",
        "avg_rfd": "Mean max RFD",
        "select_event": "Select strike",
        "event_label": "Strike",
        "peak_force": "Peak Force",
        "net_force": "Net Peak Force",
        "time_to_peak": "Time to Peak",
        "contact_dur": "Contact Duration",
        "impulse_total": "Total Impulse",
        "impulse_peak": "Impulse to Peak",
        "rfd_max": "Max RFD",
        "jerk_max": "Max Jerk",
        "f_10ms": "Force @ 10ms",
        "rfd_10": "RFD 0–10ms",
        "f_20ms": "Force @ 20ms",
        "rfd_20": "RFD 0–20ms",
        "vel_impact": "Impact Velocity",
        "vel_max": "Max Velocity",
        "acc_impact": "Acceleration at Impact",
        "strike_dur": "Strike Duration (kin.)",
        "full_metrics": "Show all metrics",
        "fig_force": "Resultant Force",
        "fig_acc": "Resultant Acceleration",
        "fig_vel": "Resultant Fist Velocity",
        "fig_vel_xyz": "Velocity by Axis",
        "fig_rfd": "Rate of Force Development (RFD)",
        "onset": "Contact onset",
        "peak": "Force peak",
        "end": "Contact end",
        "acc_onset": "Strike onset",
        "compare_title": "Strike Comparison",
        "summary_table": "Summary Table",
        "export_title": "Download Results",
        "download_excel": "⬇️ Download Excel results",
        "download_raw": "⬇️ Strike data",
        "upload_prompt": "👆 Upload an Excel file in the sidebar to get started",
        "upload_prompt_sub": "Noraxon format: columns Time, 1x/1y/1z, fx/fy/fz",
        "time_axis": "Time (ms)",
        "full_time": "Time (s)",
        "vel_impact_card": "Impact Velocity",
        "comparison_note": "Values at t = force peak time",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
MILLI_G_TO_MPS2 = 9.80665 / 1000
EVENT_COLORS = ["#E63946", "#F4A261", "#2A9D8F", "#457B9D", "#9B5DE5"]

# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def find_top_peaks(data, num_peaks, force_thresh, min_time_sep):
    valid = data[data["resultant_force"] >= force_thresh]
    sorted_data = valid.sort_values("resultant_force", ascending=False)
    peaks = []
    for idx, row in sorted_data.iterrows():
        if len(peaks) >= num_peaks:
            break
        diffs = [abs(row["Time"] - data.loc[p, "Time"]) for p in peaks]
        if all(d >= min_time_sep for d in diffs):
            peaks.append(idx)
    peaks.sort(key=lambda i: data.loc[i, "Time"])
    return peaks

def integrate_velocity(acc_values, times):
    vel = np.zeros(len(acc_values))
    for i in range(1, len(vel)):
        vel[i] = vel[i - 1] + acc_values[i - 1] * (times[i] - times[i - 1])
    return vel

def find_strike_onset(force_series, peak_idx_local, threshold):
    for i in range(peak_idx_local, -1, -1):
        if force_series.iloc[i] < threshold:
            return i
    return 0

def find_strike_end(force_series, peak_idx_local, threshold):
    for i in range(peak_idx_local, len(force_series)):
        if force_series.iloc[i] < threshold:
            return i
    return len(force_series) - 1

def find_acc_onset(acc_series, threshold):
    for i, val in enumerate(acc_series):
        if val > threshold:
            return i
    return 0

@st.cache_data(show_spinner=False)
def run_full_analysis(file_bytes, num_events, force_thresh, min_sep, window_size, acc_thresh, onset_thresh):
    # Load and preprocess inline (avoid nested @st.cache_data calls)
    raw = pd.read_excel(io.BytesIO(file_bytes))
    raw[["1x", "1y", "1z"]] = raw[["1x", "1y", "1z"]] * MILLI_G_TO_MPS2
    raw["resultant_acceleration"] = np.sqrt(raw["1x"]**2 + raw["1y"]**2 + raw["1z"]**2)
    raw["resultant_force"] = np.sqrt(raw["fx"]**2 + raw["fy"]**2 + raw["fz"]**2)
    peak_indices = find_top_peaks(raw, num_events, force_thresh, min_sep)
    all_results = []
    event_dfs = {}

    for ev_num, peak_idx in enumerate(peak_indices, 1):
        peak_time = raw.loc[peak_idx, "Time"]
        ev = raw[(raw["Time"] >= peak_time - window_size / 2) &
                 (raw["Time"] <= peak_time + window_size / 2)].copy()
        ev.reset_index(drop=True, inplace=True)
        ev["Time_rel"] = ev["Time"] - (peak_time - window_size / 2)

        times = ev["Time"].values
        ev["vel_x"] = integrate_velocity(ev["1x"].values, times)
        ev["vel_y"] = integrate_velocity(ev["1y"].values, times)
        ev["vel_z"] = integrate_velocity(ev["1z"].values, times)
        ev["resultant_velocity"] = np.sqrt(ev["vel_x"]**2 + ev["vel_y"]**2 + ev["vel_z"]**2)

        lpf = ev["resultant_force"].idxmax()
        ao  = find_acc_onset(ev["resultant_acceleration"], acc_thresh)
        fo  = find_strike_onset(ev["resultant_force"], lpf, onset_thresh)
        fe  = find_strike_end(ev["resultant_force"], lpf, onset_thresh)

        pf   = ev.loc[lpf, "resultant_force"]
        fov  = ev.loc[fo, "resultant_force"]
        tfo  = ev.loc[fo, "Time"]
        tfe  = ev.loc[fe, "Time"]
        tpf  = ev.loc[lpf, "Time"]
        tao  = ev.loc[ao, "Time"]

        ttp_ms = (tpf - tfo) * 1000
        dur_ms = (tfe - tfo) * 1000

        cd = ev.loc[fo:fe]
        imp_tot  = float(np.trapz(cd["resultant_force"], x=cd["Time"]))
        imp_peak = float(np.trapz(ev.loc[fo:lpf, "resultant_force"], x=ev.loc[fo:lpf, "Time"]))

        dt_mean = ev["Time"].diff().mean()
        ev["RFD"]  = ev["resultant_force"].diff() / dt_mean
        ev["Jerk"] = ev["RFD"].diff() / dt_mean
        max_rfd  = float(ev["RFD"].max())
        max_jerk = float(ev["Jerk"].max())

        def fat(ms):
            tgt = tfo + ms / 1000.0
            idx = (ev["Time"] - tgt).abs().idxmin()
            return float(ev.loc[idx, "resultant_force"])

        f10 = fat(10); f20 = fat(20)
        rfd10 = (f10 - fov) / 0.010
        rfd20 = (f20 - fov) / 0.020

        vp  = float(ev.loc[lpf, "resultant_velocity"])
        vmax = float(ev["resultant_velocity"].max())
        ap  = float(ev.loc[lpf, "resultant_acceleration"])

        all_results.append({
            "event_num": ev_num,
            "peak_time": round(peak_time, 3),
            "peak_force": round(pf, 2),
            "net_peak_force": round(pf - fov, 2),
            "time_to_peak_ms": round(ttp_ms, 1),
            "contact_dur_ms": round(dur_ms, 1),
            "impulse_total": round(imp_tot, 3),
            "impulse_to_peak": round(imp_peak, 3),
            "rfd_max": round(max_rfd, 0),
            "jerk_max": round(max_jerk, 0),
            "f_10ms": round(f10, 2),
            "rfd_0_10": round(rfd10, 0),
            "f_20ms": round(f20, 2),
            "rfd_0_20": round(rfd20, 0),
            "vel_at_peak": round(vp, 3),
            "vel_max": round(vmax, 3),
            "acc_at_peak": round(ap, 1),
            "strike_dur_ms": round((tpf - tao) * 1000, 1),
            # internal indices for plotting
            "_lpf": lpf, "_ao": ao, "_fo": fo, "_fe": fe,
        })
        event_dfs[ev_num] = ev

    return raw, peak_indices, all_results, event_dfs


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🥊 Boxing Biomechanics Lab")
    st.markdown("---")

    lang = st.radio("🌐 Idioma / Language", ["es", "en"],
                    format_func=lambda x: "🇪🇸 Español" if x == "es" else "🇬🇧 English",
                    horizontal=True)
    t = LANG[lang]

    st.markdown("---")
    uploaded = st.file_uploader(t["upload_label"], type=["xlsx"], help=t["upload_help"])
    athlete = st.text_input(t["athlete_label"], placeholder=t["athlete_placeholder"])

    st.markdown("---")
    with st.expander(f"⚙️ {t['params_title']}", expanded=False):
        num_events   = st.slider(t["num_events"],    min_value=1,   max_value=10,  value=5)
        force_thresh = st.slider(t["force_thresh"],  min_value=50,  max_value=800, value=300, step=25)
        min_sep      = st.slider(t["min_sep"],       min_value=0.2, max_value=2.0, value=0.5, step=0.1)
        window_size  = st.slider(t["window"],        min_value=0.2, max_value=1.0, value=0.4, step=0.05)
        acc_thresh   = st.slider(t["acc_thresh"],    min_value=5,   max_value=50,  value=12)
        onset_thresh = st.slider(t["onset_thresh"],  min_value=5,   max_value=100, value=20, step=5)

    st.markdown("---")
    analyze = st.button(t["analyze_btn"], use_container_width=True, type="primary")

# ─────────────────────────────────────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────────────────────────────────────
athlete_display = athlete if athlete else ("Atleta" if lang == "es" else "Athlete")
st.markdown(f"""
<div class="hero">
    <h1>🥊 {t['title']}</h1>
    <p>{t['subtitle']} &nbsp;|&nbsp; {athlete_display}</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────────────────────────────────────
if uploaded is None:
    st.markdown(f"""
    <div style='text-align:center; padding:4rem 2rem;'>
        <div style='font-size:5rem;'>🥊</div>
        <h2 style='color:#C62828; font-weight:800;'>{t["upload_prompt"]}</h2>
        <p style='color:#666; font-size:1rem;'>{t["upload_prompt_sub"]}</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Run analysis ──────────────────────────────────────────────────────────────
file_bytes = uploaded.read()
with st.spinner("Procesando..." if lang == "es" else "Processing..."):
    raw, peak_indices, all_results, event_dfs = run_full_analysis(
        file_bytes, num_events, force_thresh, min_sep,
        window_size, acc_thresh, onset_thresh
    )

n = len(all_results)
avg_force = np.mean([r["peak_force"] for r in all_results])
avg_vel   = np.mean([r["vel_at_peak"] for r in all_results])
avg_rfd   = np.mean([r["rfd_max"] for r in all_results])

# ── KPI bar ──────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
kpi_style = "background:linear-gradient(135deg,#1A1A2E,#16213E);border-radius:12px;padding:1rem 1.2rem;color:white;text-align:center;border-left:4px solid #C62828"
with c1:
    st.markdown(f"<div style='{kpi_style}'><div style='font-size:.7rem;opacity:.7;text-transform:uppercase;letter-spacing:1px'>{'Golpes' if lang=='es' else 'Strikes'}</div><div style='font-size:2.4rem;font-weight:800;color:#F9A825'>{n}</div><div style='font-size:.75rem;opacity:.7'>{t['detected']}</div></div>", unsafe_allow_html=True)
with c2:
    st.markdown(f"<div style='{kpi_style}'><div style='font-size:.7rem;opacity:.7;text-transform:uppercase;letter-spacing:1px'>{t['avg_force']}</div><div style='font-size:2.4rem;font-weight:800;color:#F9A825'>{avg_force:.0f}</div><div style='font-size:.75rem;opacity:.7'>N</div></div>", unsafe_allow_html=True)
with c3:
    st.markdown(f"<div style='{kpi_style}'><div style='font-size:.7rem;opacity:.7;text-transform:uppercase;letter-spacing:1px'>{t['avg_vel']}</div><div style='font-size:2.4rem;font-weight:800;color:#F9A825'>{avg_vel:.2f}</div><div style='font-size:.75rem;opacity:.7'>m/s</div></div>", unsafe_allow_html=True)
with c4:
    st.markdown(f"<div style='{kpi_style}'><div style='font-size:.7rem;opacity:.7;text-transform:uppercase;letter-spacing:1px'>{t['avg_rfd']}</div><div style='font-size:2.4rem;font-weight:800;color:#F9A825'>{avg_rfd/1000:.0f}k</div><div style='font-size:.75rem;opacity:.7'>N/s</div></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([t["tab_signal"], t["tab_events"], t["tab_compare"], t["tab_export"]])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — FULL SIGNAL
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    # Force signal
    fig_sig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.08,
                            subplot_titles=[t["fig_force"], t["fig_acc"]])

    fig_sig.add_trace(go.Scatter(
        x=raw["Time"], y=raw["resultant_force"],
        name=t["fig_force"], line=dict(color="#457B9D", width=1),
        fill="tozeroy", fillcolor="rgba(69,123,157,0.1)"
    ), row=1, col=1)

    for i, (peak_idx, res) in enumerate(zip(peak_indices, all_results)):
        c = EVENT_COLORS[i % len(EVENT_COLORS)]
        pt = raw.loc[peak_idx, "Time"]
        pf = raw.loc[peak_idx, "resultant_force"]
        # shaded window
        fig_sig.add_vrect(x0=pt - window_size/2, x1=pt + window_size/2,
                          fillcolor=c, opacity=0.12, layer="below",
                          line_width=0, row=1, col=1)
        # peak marker
        fig_sig.add_trace(go.Scatter(
            x=[pt], y=[pf],
            mode="markers+text",
            marker=dict(symbol="x", size=14, color=c, line=dict(width=2)),
            text=[f"E{i+1}<br>{pf:.0f}N"],
            textposition="top center",
            textfont=dict(size=10, color=c),
            name=f"{'Golpe' if lang=='es' else 'Strike'} {i+1}",
            showlegend=True,
        ), row=1, col=1)

    fig_sig.add_trace(go.Scatter(
        x=raw["Time"], y=raw["resultant_acceleration"],
        name=t["fig_acc"], line=dict(color="#F4A261", width=1),
        fill="tozeroy", fillcolor="rgba(244,162,97,0.1)"
    ), row=2, col=1)

    fig_sig.update_layout(
        height=550, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=20, t=60, b=40),
        font=dict(family="Inter"),
    )
    fig_sig.update_xaxes(title_text=t["full_time"], row=2, col=1)
    fig_sig.update_yaxes(title_text="N", row=1, col=1)
    fig_sig.update_yaxes(title_text="m/s²", row=2, col=1)
    st.plotly_chart(fig_sig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PER EVENT
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    ev_options = [f"{t['event_label']} {r['event_num']}  —  {r['peak_force']:.0f} N  |  {r['vel_at_peak']:.2f} m/s" for r in all_results]
    sel_idx = st.selectbox(t["select_event"], range(len(ev_options)), format_func=lambda i: ev_options[i])
    res = all_results[sel_idx]
    ev  = event_dfs[res["event_num"]]
    c   = EVENT_COLORS[sel_idx % len(EVENT_COLORS)]

    lpf = res["_lpf"]; ao = res["_ao"]; fo = res["_fo"]; fe = res["_fe"]
    t_ms = ev["Time_rel"] * 1000

    # KPI cards
    k1, k2, k3, k4 = st.columns(4)
    card = lambda label, val, unit, gold=False: f"""
    <div class='metric-card {"metric-card-gold" if gold else ""}'>
        <div class='label'>{label}</div>
        <div class='value'>{val}</div>
        <div class='unit'>{unit}</div>
    </div>"""
    with k1: st.markdown(card(t["peak_force"], f"{res['peak_force']:.0f}", "N"), unsafe_allow_html=True)
    with k2: st.markdown(card(t["vel_impact_card"], f"{res['vel_at_peak']:.2f}", "m/s", gold=True), unsafe_allow_html=True)
    with k3: st.markdown(card(t["rfd_max"], f"{res['rfd_max']/1000:.0f}k", "N/s"), unsafe_allow_html=True)
    with k4: st.markdown(card(t["impulse_total"], f"{res['impulse_total']:.2f}", "N·s"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Subplot 2x2
    fig_ev = make_subplots(
        rows=2, cols=2,
        subplot_titles=[t["fig_force"], t["fig_acc"], t["fig_vel"], t["fig_vel_xyz"]],
        vertical_spacing=0.14, horizontal_spacing=0.1
    )

    def add_vlines(fig, row, col):
        fig.add_vline(x=float(t_ms.iloc[fo]), line_dash="dash", line_color="green",  line_width=1.5, row=row, col=col, annotation_text=t["onset"], annotation_position="top left", annotation_font_size=9)
        fig.add_vline(x=float(t_ms.iloc[lpf]), line_dash="dash", line_color="#C62828", line_width=2,   row=row, col=col, annotation_text=t["peak"],  annotation_position="top right", annotation_font_size=9)
        fig.add_vline(x=float(t_ms.iloc[fe]), line_dash="dash", line_color="black",  line_width=1.5, row=row, col=col, annotation_text=t["end"],   annotation_position="top left", annotation_font_size=9)

    # Force
    fig_ev.add_trace(go.Scatter(x=t_ms, y=ev["resultant_force"],
        fill="tozeroy", fillcolor=f"rgba(69,123,157,0.15)",
        line=dict(color="#457B9D", width=2), name=t["fig_force"], showlegend=False), row=1, col=1)
    add_vlines(fig_ev, 1, 1)

    # Acceleration
    fig_ev.add_trace(go.Scatter(x=t_ms, y=ev["resultant_acceleration"],
        fill="tozeroy", fillcolor="rgba(244,162,97,0.15)",
        line=dict(color="#F4A261", width=2), name=t["fig_acc"], showlegend=False), row=1, col=2)
    fig_ev.add_vline(x=float(t_ms.iloc[ao]),  line_dash="dash", line_color="purple", line_width=1.5, row=1, col=2, annotation_text=t["acc_onset"], annotation_position="top left", annotation_font_size=9)
    fig_ev.add_vline(x=float(t_ms.iloc[lpf]), line_dash="dash", line_color="#C62828", line_width=2,  row=1, col=2)

    # Resultant velocity
    fig_ev.add_trace(go.Scatter(x=t_ms, y=ev["resultant_velocity"],
        line=dict(color="#9B5DE5", width=2.5), name=t["fig_vel"], showlegend=False), row=2, col=1)
    fig_ev.add_vline(x=float(t_ms.iloc[lpf]), line_dash="dash", line_color="#C62828", line_width=2, row=2, col=1)
    fig_ev.add_trace(go.Scatter(
        x=[float(t_ms.iloc[lpf])], y=[res["vel_at_peak"]],
        mode="markers+text",
        marker=dict(color="#C62828", size=10),
        text=[f"  {res['vel_at_peak']:.2f} m/s"],
        textposition="middle right", textfont=dict(size=11, color="#C62828", family="Inter"),
        showlegend=False
    ), row=2, col=1)

    # Velocity XYZ
    for axis, col_name, color in [("X", "vel_x", "#E63946"), ("Y", "vel_y", "#2A9D8F"), ("Z", "vel_z", "#457B9D")]:
        fig_ev.add_trace(go.Scatter(x=t_ms, y=ev[col_name],
            line=dict(color=color, width=1.8), name=axis, showlegend=True,
            legendgroup=f"xyz_{sel_idx}"), row=2, col=2)
    fig_ev.add_vline(x=float(t_ms.iloc[lpf]), line_dash="dash", line_color="#C62828", line_width=1.5, opacity=0.6, row=2, col=2)

    fig_ev.update_layout(
        height=560, template="plotly_white",
        title_text=f"{t['event_label']} {res['event_num']}  ·  t = {res['peak_time']}s",
        title_font=dict(size=14, family="Inter"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="right", x=1),
        margin=dict(l=60, r=20, t=60, b=40),
        font=dict(family="Inter"),
    )
    for r_, c_, unit_ in [(1,1,"N"), (1,2,"m/s²"), (2,1,"m/s"), (2,2,"m/s")]:
        fig_ev.update_yaxes(title_text=unit_, row=r_, col=c_)
        fig_ev.update_xaxes(title_text=t["time_axis"], row=r_, col=c_)

    st.plotly_chart(fig_ev, use_container_width=True)

    # Full metrics table (expander)
    with st.expander(f"📋 {t['full_metrics']}"):
        metrics_data = {
            t["peak_force"]:     (f"{res['peak_force']:.1f}",    "N"),
            t["net_force"]:      (f"{res['net_peak_force']:.1f}", "N"),
            t["time_to_peak"]:   (f"{res['time_to_peak_ms']:.1f}","ms"),
            t["contact_dur"]:    (f"{res['contact_dur_ms']:.1f}", "ms"),
            t["impulse_total"]:  (f"{res['impulse_total']:.3f}",  "N·s"),
            t["impulse_peak"]:   (f"{res['impulse_to_peak']:.3f}","N·s"),
            t["rfd_max"]:        (f"{res['rfd_max']:.0f}",        "N/s"),
            t["jerk_max"]:       (f"{res['jerk_max']:.0f}",       "N/s²"),
            t["f_10ms"]:         (f"{res['f_10ms']:.1f}",         "N"),
            t["rfd_10"]:         (f"{res['rfd_0_10']:.0f}",       "N/s"),
            t["f_20ms"]:         (f"{res['f_20ms']:.1f}",         "N"),
            t["rfd_20"]:         (f"{res['rfd_0_20']:.0f}",       "N/s"),
            t["vel_impact"]:     (f"{res['vel_at_peak']:.3f}",    "m/s"),
            t["vel_max"]:        (f"{res['vel_max']:.3f}",        "m/s"),
            t["acc_impact"]:     (f"{res['acc_at_peak']:.1f}",    "m/s²"),
            t["strike_dur"]:     (f"{res['strike_dur_ms']:.1f}",  "ms"),
        }
        mdf = pd.DataFrame([(k, v[0], v[1]) for k, v in metrics_data.items()],
                           columns=["Métrica" if lang=="es" else "Metric",
                                    "Valor" if lang=="es" else "Value",
                                    "Unidad" if lang=="es" else "Unit"])
        st.dataframe(mdf, use_container_width=True, hide_index=True,
                     column_config={
                         "Valor" if lang=="es" else "Value": st.column_config.TextColumn(width="small"),
                         "Unidad" if lang=="es" else "Unit": st.column_config.TextColumn(width="small"),
                     })

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(f"<div class='section-title'>{t['compare_title']}</div>", unsafe_allow_html=True)

    ev_labels = [f"E{r['event_num']}" for r in all_results]
    colors_bar = EVENT_COLORS[:len(all_results)]

    compare_metrics = [
        (t["peak_force"],   "peak_force",     "N",    "#457B9D"),
        (t["vel_impact"],   "vel_at_peak",    "m/s",  "#F4A261"),
        (t["rfd_max"],      "rfd_max",        "N/s",  "#C62828"),
        (t["impulse_total"],"impulse_total",  "N·s",  "#2A9D8F"),
        (t["time_to_peak"], "time_to_peak_ms","ms",   "#9B5DE5"),
        (t["contact_dur"],  "contact_dur_ms", "ms",   "#6D6875"),
    ]

    cols = st.columns(3)
    for idx, (label, key, unit, bcolor) in enumerate(compare_metrics):
        vals = [r[key] for r in all_results]
        fig_bar = go.Figure(go.Bar(
            x=ev_labels, y=vals,
            marker_color=[EVENT_COLORS[i % len(EVENT_COLORS)] for i in range(len(vals))],
            text=[f"{v:.1f}" for v in vals],
            textposition="outside",
            textfont=dict(size=12, family="Inter", color="#1A1A2E"),
        ))
        fig_bar.update_layout(
            title=dict(text=f"<b>{label}</b> ({unit})", font=dict(size=13, family="Inter")),
            height=280,
            template="plotly_white",
            yaxis=dict(title=unit, showgrid=True, gridcolor="#eee"),
            xaxis=dict(showgrid=False),
            margin=dict(l=40, r=20, t=50, b=30),
            font=dict(family="Inter"),
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
        )
        # add mean line
        mean_val = np.mean(vals)
        fig_bar.add_hline(y=mean_val, line_dash="dot", line_color="#999",
                          annotation_text=f"μ={mean_val:.1f}", annotation_position="right",
                          annotation_font_size=10)
        with cols[idx % 3]:
            st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown(f"<div class='section-title'>{t['summary_table']}</div>", unsafe_allow_html=True)
    sum_df = pd.DataFrame([{
        f"{'Golpe' if lang=='es' else 'Strike'}": f"E{r['event_num']}",
        "t (s)": r["peak_time"],
        t["peak_force"] + " (N)": r["peak_force"],
        t["vel_impact"] + " (m/s)": r["vel_at_peak"],
        t["rfd_max"] + " (N/s)": r["rfd_max"],
        t["time_to_peak"] + " (ms)": r["time_to_peak_ms"],
        t["contact_dur"] + " (ms)": r["contact_dur_ms"],
        t["impulse_total"] + " (N·s)": r["impulse_total"],
    } for r in all_results])
    st.dataframe(sum_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — EXPORT
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown(f"<div class='section-title'>{t['export_title']}</div>", unsafe_allow_html=True)

    # Build Excel
    output = io.BytesIO()
    col_names = {
        "event_num":        "Evento" if lang=="es" else "Event",
        "peak_time":        "Tiempo Pico (s)" if lang=="es" else "Peak Time (s)",
        "peak_force":       t["peak_force"] + " (N)",
        "net_peak_force":   t["net_force"] + " (N)",
        "time_to_peak_ms":  t["time_to_peak"] + " (ms)",
        "contact_dur_ms":   t["contact_dur"] + " (ms)",
        "impulse_total":    t["impulse_total"] + " (N·s)",
        "impulse_to_peak":  t["impulse_peak"] + " (N·s)",
        "rfd_max":          t["rfd_max"] + " (N/s)",
        "jerk_max":         t["jerk_max"] + " (N/s²)",
        "f_10ms":           t["f_10ms"] + " (N)",
        "rfd_0_10":         t["rfd_10"] + " (N/s)",
        "f_20ms":           t["f_20ms"] + " (N)",
        "rfd_0_20":         t["rfd_20"] + " (N/s)",
        "vel_at_peak":      t["vel_impact"] + " (m/s)",
        "vel_max":          t["vel_max"] + " (m/s)",
        "acc_at_peak":      t["acc_impact"] + " (m/s²)",
        "strike_dur_ms":    t["strike_dur"] + " (ms)",
    }
    export_keys = [k for k in col_names if not k.startswith("_")]
    rows = [{col_names[k]: r[k] for k in export_keys} for r in all_results]
    summary_export = pd.DataFrame(rows)

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        summary_export.to_excel(writer, sheet_name="Resumen" if lang=="es" else "Summary", index=False)
        pd.DataFrame([{col_names[k]: round(np.mean([r[k] for r in all_results]), 3) for k in export_keys},
                      {col_names[k]: round(np.std ([r[k] for r in all_results]), 3) for k in export_keys}],
                     index=["Media/Mean", "SD"]).to_excel(writer, sheet_name="Stats", index=True)
        for ev_num, ev_df in event_dfs.items():
            ev_df.drop(columns=["RFD", "Jerk"], errors="ignore").to_excel(
                writer, sheet_name=f"{'Golpe' if lang=='es' else 'Strike'}_{ev_num}", index=False)
    output.seek(0)

    fname = f"{athlete_display}_resultados.xlsx" if lang=="es" else f"{athlete_display}_results.xlsx"
    st.download_button(
        label=t["download_excel"],
        data=output,
        file_name=fname,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
        type="primary",
    )

    st.markdown(f"<div class='section-title'>{t['summary_table']}</div>", unsafe_allow_html=True)
    st.dataframe(summary_export, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown(f"**{'Descargar datos por golpe:' if lang=='es' else 'Download data by strike:'}**")
    dl_cols = st.columns(len(all_results))
    for i, (ev_num, ev_df) in enumerate(event_dfs.items()):
        buf = io.BytesIO()
        ev_df.drop(columns=["RFD", "Jerk"], errors="ignore").to_excel(buf, index=False)
        buf.seek(0)
        with dl_cols[i]:
            st.download_button(
                label=f"{t['download_raw']} E{ev_num}",
                data=buf,
                file_name=f"{athlete_display}_E{ev_num}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
