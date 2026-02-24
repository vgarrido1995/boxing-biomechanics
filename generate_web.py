"""
Generates docs/index.html — a standalone interactive web page
with all boxing biomechanics analysis for K001_J.
Run locally: python generate_web.py
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import json, os

# ── Config ────────────────────────────────────────────────────────────────────
FILENAME        = "K001_J left hand pad.xlsx"
MILLI_G_TO_MPS2 = 9.80665 / 1000
FORCE_THRESHOLD = 300
MIN_TIME_SEP    = 0.5
WINDOW_SIZE     = 0.4
ACC_THRESHOLD   = 12
ONSET_THRESHOLD = 20
NUM_EVENTS      = 5
EVENT_COLORS    = ["#E63946", "#F4A261", "#2A9D8F", "#457B9D", "#9B5DE5"]
ATHLETE         = "K001_J"

# ── Analysis functions ────────────────────────────────────────────────────────
def find_top_peaks(data):
    valid = data[data["resultant_force"] >= FORCE_THRESHOLD]
    sorted_data = valid.sort_values("resultant_force", ascending=False)
    peaks = []
    for idx, row in sorted_data.iterrows():
        if len(peaks) >= NUM_EVENTS: break
        diffs = [abs(row["Time"] - data.loc[p, "Time"]) for p in peaks]
        if all(d >= MIN_TIME_SEP for d in diffs):
            peaks.append(idx)
    peaks.sort(key=lambda i: data.loc[i, "Time"])
    return peaks

def integrate_velocity(acc, times):
    vel = np.zeros(len(acc))
    for i in range(1, len(vel)):
        vel[i] = vel[i-1] + acc[i-1] * (times[i] - times[i-1])
    return vel

def strike_onset(fs, pk, thr=ONSET_THRESHOLD):
    for i in range(pk, -1, -1):
        if fs.iloc[i] < thr: return i
    return 0

def strike_end(fs, pk, thr=ONSET_THRESHOLD):
    for i in range(pk, len(fs)):
        if fs.iloc[i] < thr: return i
    return len(fs) - 1

def acc_onset(acc, thr=ACC_THRESHOLD):
    for i, v in enumerate(acc):
        if v > thr: return i
    return 0

# ── Load & preprocess ─────────────────────────────────────────────────────────
print("Loading data...")
raw = pd.read_excel(FILENAME)
raw[["1x","1y","1z"]] *= MILLI_G_TO_MPS2
raw["resultant_acceleration"] = np.sqrt(raw["1x"]**2 + raw["1y"]**2 + raw["1z"]**2)
raw["resultant_force"]        = np.sqrt(raw["fx"]**2  + raw["fy"]**2  + raw["fz"]**2)

# ── Detect events ─────────────────────────────────────────────────────────────
peak_indices = find_top_peaks(raw)
print(f"Events found: {len(peak_indices)}")

# ── Per-event analysis ────────────────────────────────────────────────────────
all_results, event_dfs = [], {}

for ev_num, peak_idx in enumerate(peak_indices, 1):
    peak_time = raw.loc[peak_idx, "Time"]
    ev = raw[(raw["Time"] >= peak_time - WINDOW_SIZE/2) &
             (raw["Time"] <= peak_time + WINDOW_SIZE/2)].copy()
    ev.reset_index(drop=True, inplace=True)
    ev["Time_rel"] = ev["Time"] - (peak_time - WINDOW_SIZE/2)

    times = ev["Time"].values
    ev["vel_x"] = integrate_velocity(ev["1x"].values, times)
    ev["vel_y"] = integrate_velocity(ev["1y"].values, times)
    ev["vel_z"] = integrate_velocity(ev["1z"].values, times)
    ev["resultant_velocity"] = np.sqrt(ev["vel_x"]**2 + ev["vel_y"]**2 + ev["vel_z"]**2)

    lpf = ev["resultant_force"].idxmax()
    ao  = acc_onset(ev["resultant_acceleration"])
    fo  = strike_onset(ev["resultant_force"], lpf)
    fe  = strike_end(ev["resultant_force"], lpf)

    pf  = ev.loc[lpf, "resultant_force"]
    fov = ev.loc[fo,  "resultant_force"]
    tfo = ev.loc[fo,  "Time"]
    tfe = ev.loc[fe,  "Time"]
    tpf = ev.loc[lpf, "Time"]
    tao = ev.loc[ao,  "Time"]

    ttp_ms = (tpf - tfo) * 1000
    dur_ms = (tfe - tfo) * 1000

    cd = ev.loc[fo:fe]
    imp_tot  = float(np.trapz(cd["resultant_force"], x=cd["Time"]))
    imp_peak = float(np.trapz(ev.loc[fo:lpf,"resultant_force"], x=ev.loc[fo:lpf,"Time"]))

    dt = ev["Time"].diff().mean()
    ev["RFD"] = ev["resultant_force"].diff() / dt
    max_rfd   = float(ev["RFD"].max())

    def fat(ms):
        tgt = tfo + ms/1000.0
        idx = (ev["Time"] - tgt).abs().idxmin()
        return float(ev.loc[idx, "resultant_force"])

    f10 = fat(10); f20 = fat(20)

    all_results.append({
        "ev": ev_num,
        "peak_time": round(peak_time, 3),
        "peak_force": round(pf, 1),
        "net_peak_force": round(pf - fov, 1),
        "time_to_peak_ms": round(ttp_ms, 1),
        "contact_dur_ms": round(dur_ms, 1),
        "impulse_total": round(imp_tot, 3),
        "impulse_to_peak": round(imp_peak, 3),
        "rfd_max": round(max_rfd, 0),
        "f_10ms": round(f10, 1),
        "rfd_0_10": round((f10-fov)/0.010, 0),
        "f_20ms": round(f20, 1),
        "rfd_0_20": round((f20-fov)/0.020, 0),
        "vel_at_peak": round(float(ev.loc[lpf,"resultant_velocity"]), 3),
        "vel_max": round(float(ev["resultant_velocity"].max()), 3),
        "acc_at_peak": round(float(ev.loc[lpf,"resultant_acceleration"]), 1),
        "strike_dur_ms": round((tpf - tao)*1000, 1),
        "_lpf": int(lpf), "_ao": int(ao), "_fo": int(fo), "_fe": int(fe),
    })
    event_dfs[ev_num] = ev
    print(f"  E{ev_num}: {pf:.0f}N  {ev.loc[lpf,'resultant_velocity']:.2f}m/s")

# ── Build Plotly figures → JSON ───────────────────────────────────────────────
PLOTLY_CFG = dict(template="plotly_white", font=dict(family="Inter, sans-serif"),
                  margin=dict(l=55, r=20, t=45, b=40))

def fig_to_json(fig):
    return pio.to_json(fig)

# 1. Full signal overview
print("Building full signal chart...")
fig_sig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=["Resultant Force (N)", "Resultant Acceleration (m/s²)"])
fig_sig.add_trace(go.Scatter(x=raw["Time"], y=raw["resultant_force"],
    line=dict(color="#457B9D", width=0.9), fill="tozeroy",
    fillcolor="rgba(69,123,157,0.1)", name="Force", showlegend=False), row=1, col=1)
fig_sig.add_trace(go.Scatter(x=raw["Time"], y=raw["resultant_acceleration"],
    line=dict(color="#F4A261", width=0.9), fill="tozeroy",
    fillcolor="rgba(244,162,97,0.1)", name="Acceleration", showlegend=False), row=2, col=1)
for i, (pi, res) in enumerate(zip(peak_indices, all_results)):
    c  = EVENT_COLORS[i]
    pt = raw.loc[pi, "Time"]
    pf_val = raw.loc[pi, "resultant_force"]
    fig_sig.add_vrect(x0=pt-WINDOW_SIZE/2, x1=pt+WINDOW_SIZE/2,
                      fillcolor=c, opacity=0.13, layer="below", line_width=0, row=1, col=1)
    fig_sig.add_trace(go.Scatter(x=[pt], y=[pf_val],
        mode="markers+text", marker=dict(symbol="x", size=13, color=c, line=dict(width=2.5)),
        text=[f"E{i+1}<br>{pf_val:.0f}N"], textposition="top center",
        textfont=dict(size=10, color=c), name=f"Strike {i+1}", showlegend=True), row=1, col=1)
fig_sig.update_layout(height=520, **PLOTLY_CFG,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
fig_sig.update_xaxes(title_text="Time (s)", row=2, col=1)
fig_sig.update_yaxes(title_text="N",    row=1, col=1)
fig_sig.update_yaxes(title_text="m/s²", row=2, col=1)
json_sig = fig_to_json(fig_sig)

# 2. Per-event figures
print("Building per-event charts...")
json_events = []
for ev_num, (res, pi) in enumerate(zip(all_results, peak_indices), 1):
    ev   = event_dfs[ev_num]
    c    = EVENT_COLORS[ev_num-1]
    t_ms = ev["Time_rel"] * 1000
    lpf  = res["_lpf"]; fo = res["_fo"]; fe = res["_fe"]; ao = res["_ao"]

    fig_ev = make_subplots(rows=2, cols=2, vertical_spacing=0.15, horizontal_spacing=0.1,
        subplot_titles=["Resultant Force (N)", "Resultant Acceleration (m/s²)",
                        "Resultant Fist Velocity (m/s)", "Velocity by Axis (m/s)"])

    # Force
    fig_ev.add_trace(go.Scatter(x=t_ms, y=ev["resultant_force"],
        fill="tozeroy", fillcolor="rgba(69,123,157,0.15)", line=dict(color="#457B9D", width=2),
        name="Force", showlegend=False), row=1, col=1)
    for xval, lbl, col_ in [(t_ms.iloc[fo],"Onset","#2A9D8F"),
                             (t_ms.iloc[lpf],"Peak","#C62828"),
                             (t_ms.iloc[fe],"End","#333333")]:
        fig_ev.add_vline(x=float(xval), line_dash="dash", line_color=col_,
                         line_width=1.5, annotation_text=lbl,
                         annotation_font_size=9, row=1, col=1)

    # Acceleration
    fig_ev.add_trace(go.Scatter(x=t_ms, y=ev["resultant_acceleration"],
        fill="tozeroy", fillcolor="rgba(244,162,97,0.15)", line=dict(color="#F4A261", width=2),
        name="Acc", showlegend=False), row=1, col=2)
    fig_ev.add_vline(x=float(t_ms.iloc[ao]),  line_dash="dash", line_color="#9B5DE5",
                     line_width=1.5, annotation_text="Onset", annotation_font_size=9, row=1, col=2)
    fig_ev.add_vline(x=float(t_ms.iloc[lpf]), line_dash="dash", line_color="#C62828",
                     line_width=1.5, row=1, col=2)

    # Velocity resultant
    fig_ev.add_trace(go.Scatter(x=t_ms, y=ev["resultant_velocity"],
        line=dict(color="#9B5DE5", width=2.5), name="Vel", showlegend=False), row=2, col=1)
    fig_ev.add_vline(x=float(t_ms.iloc[lpf]), line_dash="dash", line_color="#C62828",
                     line_width=2, row=2, col=1)
    fig_ev.add_trace(go.Scatter(
        x=[float(t_ms.iloc[lpf])], y=[res["vel_at_peak"]],
        mode="markers+text", marker=dict(color="#C62828", size=10),
        text=[f"  {res['vel_at_peak']:.2f} m/s"], textposition="middle right",
        textfont=dict(size=11, color="#C62828"), showlegend=False), row=2, col=1)

    # Velocity XYZ
    for ax, col_name, col_c in [("X","vel_x","#E63946"),("Y","vel_y","#2A9D8F"),("Z","vel_z","#457B9D")]:
        fig_ev.add_trace(go.Scatter(x=t_ms, y=ev[col_name],
            line=dict(color=col_c, width=1.8), name=ax, showlegend=True), row=2, col=2)
    fig_ev.add_vline(x=float(t_ms.iloc[lpf]), line_dash="dash", line_color="#C62828",
                     line_width=1, opacity=0.5, row=2, col=2)

    fig_ev.update_layout(height=520, **PLOTLY_CFG,
        title_text=f"Strike {ev_num}  ·  t={res['peak_time']}s  ·  Peak Force: {res['peak_force']:.0f} N  ·  Impact Velocity: {res['vel_at_peak']:.2f} m/s",
        title_font=dict(size=13),
        legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="right", x=1))
    for r_, c_, u_ in [(1,1,"N"),(1,2,"m/s²"),(2,1,"m/s"),(2,2,"m/s")]:
        fig_ev.update_yaxes(title_text=u_, row=r_, col=c_)
        fig_ev.update_xaxes(title_text="Time (ms)", row=r_, col=c_)

    json_events.append(fig_to_json(fig_ev))

# 3. Comparison chart
print("Building comparison chart...")
compare_metrics = [
    ("Peak Force",        "peak_force",       "N",    "#457B9D"),
    ("Impact Velocity",   "vel_at_peak",       "m/s",  "#F4A261"),
    ("Max RFD",           "rfd_max",           "N/s",  "#C62828"),
    ("Total Impulse",     "impulse_total",     "N·s",  "#2A9D8F"),
    ("Time to Peak",      "time_to_peak_ms",   "ms",   "#9B5DE5"),
    ("Contact Duration",  "contact_dur_ms",    "ms",   "#6D6875"),
]
fig_cmp = make_subplots(rows=2, cols=3, vertical_spacing=0.18, horizontal_spacing=0.1,
    subplot_titles=[f"{m[0]} ({m[2]})" for m in compare_metrics])
ev_labels = [f"E{r['ev']}" for r in all_results]
for idx, (label, key, unit, _) in enumerate(compare_metrics):
    r_, c_ = divmod(idx, 3)
    vals = [r[key] for r in all_results]
    mean_v = np.mean(vals)
    fig_cmp.add_trace(go.Bar(
        x=ev_labels, y=vals,
        marker_color=EVENT_COLORS[:len(vals)],
        text=[f"{v:.1f}" for v in vals],
        textposition="outside", textfont=dict(size=11),
        showlegend=False, name=label), row=r_+1, col=c_+1)
    fig_cmp.add_hline(y=mean_v, line_dash="dot", line_color="#999",
                      annotation_text=f"μ={mean_v:.1f}", annotation_font_size=9,
                      row=r_+1, col=c_+1)
    fig_cmp.update_yaxes(title_text=unit, row=r_+1, col=c_+1)
fig_cmp.update_layout(height=560, **PLOTLY_CFG)
json_cmp = fig_to_json(fig_cmp)

# ── Summary table data ────────────────────────────────────────────────────────
summary_rows = []
for r in all_results:
    summary_rows.append([
        f"E{r['ev']}", r["peak_time"],
        r["peak_force"], r["vel_at_peak"], r["rfd_max"],
        r["time_to_peak_ms"], r["contact_dur_ms"],
        r["impulse_total"], r["f_10ms"], r["f_20ms"],
        r["acc_at_peak"], r["strike_dur_ms"],
    ])

# ── Embed everything in HTML ──────────────────────────────────────────────────
print("Building HTML page...")

avg_force = np.mean([r["peak_force"]  for r in all_results])
avg_vel   = np.mean([r["vel_at_peak"] for r in all_results])
avg_rfd   = np.mean([r["rfd_max"]     for r in all_results])

kpi_cards = ""
for i, r in enumerate(all_results):
    c = EVENT_COLORS[i]
    kpi_cards += f"""
    <div class="col">
      <div class="kpi-card" style="border-left:4px solid {c}">
        <div class="kpi-label">Strike {r['ev']} &nbsp; <span style="font-size:.7rem;opacity:.6">t={r['peak_time']}s</span></div>
        <div class="kpi-value" style="color:{c}">{r['peak_force']:.0f} <span class="kpi-unit">N</span></div>
        <div class="kpi-sub">{r['vel_at_peak']:.2f} m/s &nbsp;|&nbsp; RFD {r['rfd_max']/1000:.0f}k N/s</div>
      </div>
    </div>"""

metric_labels = [
    "Peak Force (N)", "Net Peak Force (N)", "Time to Peak (ms)", "Contact Duration (ms)",
    "Total Impulse (N·s)", "Impulse to Peak (N·s)", "Max RFD (N/s)",
    "Force @ 10ms (N)", "RFD 0-10ms (N/s)", "Force @ 20ms (N)", "RFD 0-20ms (N/s)",
    "Impact Velocity (m/s)", "Max Velocity (m/s)", "Acceleration at Peak (m/s²)", "Strike Duration kin. (ms)"
]
metric_keys = [
    "peak_force","net_peak_force","time_to_peak_ms","contact_dur_ms",
    "impulse_total","impulse_to_peak","rfd_max",
    "f_10ms","rfd_0_10","f_20ms","rfd_0_20",
    "vel_at_peak","vel_max","acc_at_peak","strike_dur_ms"
]

event_tabs_btns = ""
event_tabs_panes = ""
for i, r in enumerate(all_results):
    active = "active" if i == 0 else ""
    c = EVENT_COLORS[i]
    event_tabs_btns += f'<button class="tab-btn {active}" onclick="showEvent({i})" id="evbtn-{i}" style="--ec:{c}">Strike {r["ev"]}<br><small>{r["peak_force"]:.0f}N</small></button>\n'

    rows_html = ""
    for lbl, key in zip(metric_labels, metric_keys):
        val = r.get(key, "")
        rows_html += f"<tr><td>{lbl}</td><td><strong>{val}</strong></td></tr>\n"

    event_tabs_panes += f"""
    <div class="event-pane {'active' if i==0 else ''}" id="evpane-{i}">
      <div id="ev-plot-{i}" class="plotly-chart"></div>
      <div class="metrics-table-wrap">
        <table class="metrics-table">
          <thead><tr><th>Metric</th><th>Value</th></tr></thead>
          <tbody>{rows_html}</tbody>
        </table>
      </div>
    </div>"""

summary_thead = "<tr><th>Strike</th><th>t (s)</th><th>Peak Force (N)</th><th>Vel. Impact (m/s)</th><th>RFD Max (N/s)</th><th>T. to Peak (ms)</th><th>Duration (ms)</th><th>Impulse (N·s)</th><th>F@10ms (N)</th><th>F@20ms (N)</th><th>Acc@Peak (m/s²)</th><th>Dur. kin (ms)</th></tr>"
summary_tbody = ""
for row in summary_rows:
    summary_tbody += "<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>\n"

HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Boxing Biomechanics Lab — {ATHLETE}</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap" rel="stylesheet">
<style>
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: 'Inter', sans-serif; background: #f4f6fa; color: #1a1a2e; }}

/* NAV */
nav {{
  background: linear-gradient(135deg, #1a1a2e 0%, #c62828 100%);
  padding: 1rem 2rem;
  display: flex; align-items: center; justify-content: space-between;
  position: sticky; top: 0; z-index: 100; box-shadow: 0 2px 12px rgba(0,0,0,.3);
}}
nav .brand {{ color: white; font-size: 1.3rem; font-weight: 800; letter-spacing: -0.5px; }}
nav .brand span {{ color: #f9a825; }}
nav .nav-links {{ display: flex; gap: 0.5rem; }}
nav .nav-links a {{
  color: rgba(255,255,255,.75); text-decoration: none; font-size: .85rem;
  font-weight: 600; padding: .4rem .9rem; border-radius: 20px;
  transition: all .2s;
}}
nav .nav-links a:hover, nav .nav-links a.active {{
  background: rgba(255,255,255,.15); color: white;
}}

/* HERO */
.hero {{
  background: linear-gradient(135deg, #1a1a2e 0%, #c62828 100%);
  padding: 3rem 2rem 2.5rem;
  text-align: center; color: white;
}}
.hero h1 {{ font-size: 2.6rem; font-weight: 800; margin-bottom: .5rem; }}
.hero p  {{ font-size: 1.05rem; opacity: .85; }}
.hero .badge {{
  display: inline-block; background: rgba(249,168,37,.2); color: #f9a825;
  border: 1px solid #f9a825; border-radius: 20px; padding: .3rem 1rem;
  font-size: .8rem; font-weight: 700; margin-top: .8rem; letter-spacing: 1px;
}}

/* MAIN */
main {{ max-width: 1280px; margin: 0 auto; padding: 2rem 1.5rem; }}

/* SECTION */
.section {{ margin-bottom: 3rem; }}
.section-title {{
  font-size: 1.15rem; font-weight: 700; color: #1a1a2e;
  border-left: 4px solid #c62828; padding-left: .75rem;
  margin-bottom: 1.2rem;
}}

/* KPI CARDS */
.kpi-grid {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 1rem; margin-bottom: 2rem; }}
.col {{ }}
.kpi-card {{
  background: linear-gradient(135deg, #1a1a2e, #16213e);
  border-radius: 12px; padding: 1.2rem 1rem; color: white; text-align: center;
  border-left: 4px solid #c62828;
}}
.kpi-label {{ font-size: .7rem; font-weight: 700; opacity: .65; text-transform: uppercase; letter-spacing: .5px; }}
.kpi-value {{ font-size: 2.1rem; font-weight: 800; margin: .3rem 0; }}
.kpi-unit  {{ font-size: 1rem; font-weight: 400; }}
.kpi-sub   {{ font-size: .75rem; opacity: .65; }}

/* AVG KPI ROW */
.avg-row {{ display: flex; gap: 1rem; margin-bottom: 2rem; }}
.avg-card {{
  flex: 1; background: white; border-radius: 10px; padding: 1rem 1.2rem;
  box-shadow: 0 2px 8px rgba(0,0,0,.06); border-top: 3px solid #c62828;
  text-align: center;
}}
.avg-card .label {{ font-size: .7rem; text-transform: uppercase; letter-spacing: .5px; color: #666; font-weight: 600; }}
.avg-card .value {{ font-size: 1.7rem; font-weight: 800; color: #c62828; margin: .2rem 0; }}
.avg-card .unit  {{ font-size: .8rem; color: #999; }}

/* PLOTLY */
.plotly-chart {{ background: white; border-radius: 12px; padding: 1rem;
  box-shadow: 0 2px 12px rgba(0,0,0,.06); margin-bottom: 1rem; }}

/* EVENT TABS */
.event-tabs-btns {{
  display: flex; gap: .5rem; margin-bottom: 1rem; flex-wrap: wrap;
}}
.tab-btn {{
  background: white; border: 2px solid #ddd; border-radius: 10px;
  padding: .5rem 1.2rem; cursor: pointer; font-family: 'Inter', sans-serif;
  font-size: .85rem; font-weight: 600; color: #555; transition: all .2s; text-align: center;
}}
.tab-btn:hover {{ border-color: var(--ec); color: var(--ec); }}
.tab-btn.active {{ background: var(--ec); border-color: var(--ec); color: white; }}

.event-pane {{ display: none; }}
.event-pane.active {{ display: block; }}

/* METRICS TABLE */
.metrics-table-wrap {{
  background: white; border-radius: 12px; overflow: hidden;
  box-shadow: 0 2px 12px rgba(0,0,0,.06); margin-top: 1rem;
}}
.metrics-table {{ width: 100%; border-collapse: collapse; font-size: .88rem; }}
.metrics-table thead {{ background: #1a1a2e; color: white; }}
.metrics-table th, .metrics-table td {{ padding: .7rem 1rem; text-align: left; border-bottom: 1px solid #f0f0f0; }}
.metrics-table tr:last-child td {{ border-bottom: none; }}
.metrics-table tr:nth-child(even) td {{ background: #fafafa; }}
.metrics-table td:last-child {{ color: #c62828; font-weight: 700; }}

/* SUMMARY TABLE */
.summary-wrap {{ background: white; border-radius: 12px; overflow-x: auto;
  box-shadow: 0 2px 12px rgba(0,0,0,.06); }}
.summary-table {{ width: 100%; border-collapse: collapse; font-size: .82rem; white-space: nowrap; }}
.summary-table thead {{ background: #1a1a2e; color: white; position: sticky; top: 0; }}
.summary-table th, .summary-table td {{ padding: .65rem 1rem; text-align: center; border-bottom: 1px solid #f0f0f0; }}
.summary-table td:first-child {{ font-weight: 800; color: #c62828; }}
.summary-table tr:hover td {{ background: #fef5f5; }}

/* PAGE SECTIONS */
.page {{ display: none; }}
.page.active {{ display: block; }}

/* FOOTER */
footer {{
  text-align: center; padding: 2rem; color: #999; font-size: .8rem;
  border-top: 1px solid #eee; margin-top: 3rem;
}}

@media (max-width: 768px) {{
  .kpi-grid {{ grid-template-columns: repeat(2, 1fr); }}
  .avg-row  {{ flex-direction: column; }}
  nav .nav-links a {{ font-size: .75rem; padding: .3rem .6rem; }}
}}
</style>
</head>
<body>

<nav>
  <div class="brand">🥊 Boxing <span>Biomechanics</span> Lab</div>
  <div class="nav-links">
    <a href="#" class="active" onclick="showPage('signal',this)">Full Signal</a>
    <a href="#" onclick="showPage('events',this)">Per Event</a>
    <a href="#" onclick="showPage('compare',this)">Comparison</a>
    <a href="#" onclick="showPage('summary',this)">Summary</a>
  </div>
</nav>

<div class="hero">
  <h1>🥊 Boxing Biomechanics Lab</h1>
  <p>Kinematic &amp; kinetic strike analysis &nbsp;·&nbsp; Noraxon IMU + Force Sensor</p>
  <div class="badge">ATHLETE: {ATHLETE} &nbsp;·&nbsp; LEFT HAND PAD &nbsp;·&nbsp; {NUM_EVENTS} STRIKES ANALYZED</div>
</div>

<main>

<!-- AVG KPI ROW -->
<div class="avg-row" style="margin-top:1.5rem">
  <div class="avg-card">
    <div class="label">Strikes Detected</div>
    <div class="value">{NUM_EVENTS}</div>
    <div class="unit">events</div>
  </div>
  <div class="avg-card">
    <div class="label">Mean Peak Force</div>
    <div class="value">{avg_force:.0f}</div>
    <div class="unit">N</div>
  </div>
  <div class="avg-card">
    <div class="label">Mean Impact Velocity</div>
    <div class="value">{avg_vel:.2f}</div>
    <div class="unit">m/s</div>
  </div>
  <div class="avg-card">
    <div class="label">Mean Max RFD</div>
    <div class="value">{avg_rfd/1000:.0f}k</div>
    <div class="unit">N/s</div>
  </div>
</div>

<!-- KPI CARDS -->
<div class="kpi-grid">
  {kpi_cards}
</div>

<!-- ═══════════ PAGE: SIGNAL ═══════════ -->
<div class="page active" id="page-signal">
  <div class="section">
    <div class="section-title">Full Signal — Resultant Force &amp; Acceleration</div>
    <div class="plotly-chart" id="plot-signal"></div>
  </div>
</div>

<!-- ═══════════ PAGE: EVENTS ═══════════ -->
<div class="page" id="page-events">
  <div class="section">
    <div class="section-title">Per-Event Analysis</div>
    <div class="event-tabs-btns">
      {event_tabs_btns}
    </div>
    {event_tabs_panes}
  </div>
</div>

<!-- ═══════════ PAGE: COMPARE ═══════════ -->
<div class="page" id="page-compare">
  <div class="section">
    <div class="section-title">Strike Comparison</div>
    <div class="plotly-chart" id="plot-compare"></div>
  </div>
</div>

<!-- ═══════════ PAGE: SUMMARY ═══════════ -->
<div class="page" id="page-summary">
  <div class="section">
    <div class="section-title">Summary Table</div>
    <div class="summary-wrap">
      <table class="summary-table">
        <thead>{summary_thead}</thead>
        <tbody>{summary_tbody}</tbody>
      </table>
    </div>
  </div>
</div>

</main>

<footer>
  Boxing Biomechanics Lab &nbsp;·&nbsp; Noraxon Data Analysis &nbsp;·&nbsp; {ATHLETE}
</footer>

<script>
// ── Chart data (embedded) ───────────────────────────────────────────────────
const SIG_DATA = {json_sig};
const CMP_DATA = {json_cmp};
const EV_DATA  = [{','.join(json_events)}];

// ── Page navigation ─────────────────────────────────────────────────────────
function showPage(name, el) {{
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-links a').forEach(a => a.classList.remove('active'));
  document.getElementById('page-' + name).classList.add('active');
  if (el) el.classList.add('active');
  // lazy-render charts
  if (name === 'signal')  renderSignal();
  if (name === 'events')  renderEvent(currentEvent);
  if (name === 'compare') renderCompare();
}}

// ── Signal chart ─────────────────────────────────────────────────────────────
let sigRendered = false;
function renderSignal() {{
  if (sigRendered) return;
  Plotly.react('plot-signal', SIG_DATA.data, SIG_DATA.layout, {{responsive: true}});
  sigRendered = true;
}}

// ── Per-event charts ──────────────────────────────────────────────────────────
let currentEvent = 0;
const evRendered = {{}};
function showEvent(i) {{
  document.querySelectorAll('.event-pane').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('evpane-' + i).classList.add('active');
  document.getElementById('evbtn-' + i).classList.add('active');
  currentEvent = i;
  renderEvent(i);
}}
function renderEvent(i) {{
  if (evRendered[i]) return;
  const d = EV_DATA[i];
  Plotly.react('ev-plot-' + i, d.data, d.layout, {{responsive: true}});
  evRendered[i] = true;
}}

// ── Comparison chart ──────────────────────────────────────────────────────────
let cmpRendered = false;
function renderCompare() {{
  if (cmpRendered) return;
  Plotly.react('plot-compare', CMP_DATA.data, CMP_DATA.layout, {{responsive: true}});
  cmpRendered = true;
}}

// ── Init ─────────────────────────────────────────────────────────────────────
window.addEventListener('load', () => {{
  renderSignal();
  renderEvent(0);
}});
</script>
</body>
</html>
"""

os.makedirs("docs", exist_ok=True)
out_path = os.path.join("docs", "index.html")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(HTML)

size_kb = os.path.getsize(out_path) / 1024
print(f"\nGenerated: {out_path}  ({size_kb:.0f} KB)")
print("Done! Push docs/index.html to GitHub and enable GitHub Pages.")
