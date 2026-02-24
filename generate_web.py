"""
Generates docs/index.html — standalone interactive boxing biomechanics report.
Run: python generate_web.py
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

# ── Analysis ──────────────────────────────────────────────────────────────────
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

def acc_onset_fn(acc, thr=ACC_THRESHOLD):
    for i, v in enumerate(acc):
        if v > thr: return i
    return 0

print("Loading data...")
raw = pd.read_excel(FILENAME)
raw[["1x","1y","1z"]] *= MILLI_G_TO_MPS2
raw["resultant_acceleration"] = np.sqrt(raw["1x"]**2 + raw["1y"]**2 + raw["1z"]**2)
raw["resultant_force"]        = np.sqrt(raw["fx"]**2  + raw["fy"]**2  + raw["fz"]**2)

peak_indices = find_top_peaks(raw)
print(f"Events: {len(peak_indices)}")

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
    ao  = acc_onset_fn(ev["resultant_acceleration"])
    fo  = strike_onset(ev["resultant_force"], lpf)
    fe  = strike_end(ev["resultant_force"], lpf)

    pf  = ev.loc[lpf, "resultant_force"]
    fov = ev.loc[fo,  "resultant_force"]
    tfo = ev.loc[fo,  "Time"]
    tfe = ev.loc[fe,  "Time"]
    tpf = ev.loc[lpf, "Time"]
    tao = ev.loc[ao,  "Time"]

    ttp_ms  = (tpf - tfo) * 1000
    dur_ms  = (tfe - tfo) * 1000
    kin_ms  = (tpf - tao) * 1000

    cd       = ev.loc[fo:fe]
    imp_tot  = float(np.trapz(cd["resultant_force"], x=cd["Time"]))
    imp_peak = float(np.trapz(ev.loc[fo:lpf,"resultant_force"], x=ev.loc[fo:lpf,"Time"]))

    dt      = ev["Time"].diff().mean()
    ev["RFD"]  = ev["resultant_force"].diff() / dt
    ev["Jerk"] = ev["RFD"].diff() / dt
    max_rfd    = float(ev["RFD"].max())
    max_jerk   = float(ev["Jerk"].max())

    def fat(ms):
        tgt = tfo + ms/1000.0
        idx = (ev["Time"] - tgt).abs().idxmin()
        return float(ev.loc[idx, "resultant_force"])

    f10 = fat(10); f20 = fat(20)

    all_results.append({
        "ev": ev_num, "peak_time": round(peak_time, 3),
        "peak_force":       round(pf, 1),
        "net_peak_force":   round(pf - fov, 1),
        "time_to_peak_ms":  round(ttp_ms, 1),
        "contact_dur_ms":   round(dur_ms, 1),
        "kin_dur_ms":       round(kin_ms, 1),
        "impulse_total":    round(imp_tot, 2),
        "impulse_to_peak":  round(imp_peak, 2),
        "rfd_max":          round(max_rfd, 0),
        "jerk_max":         round(max_jerk, 0),
        "f_10ms":           round(f10, 1),
        "rfd_0_10":         round((f10-fov)/0.010, 0),
        "f_20ms":           round(f20, 1),
        "rfd_0_20":         round((f20-fov)/0.020, 0),
        "vel_at_peak":      round(float(ev.loc[lpf,"resultant_velocity"]), 2),
        "vel_max":          round(float(ev["resultant_velocity"].max()), 2),
        "acc_at_peak":      round(float(ev.loc[lpf,"resultant_acceleration"]), 1),
        "acc_max":          round(float(ev["resultant_acceleration"].max()), 1),
        "_lpf": int(lpf), "_ao": int(ao), "_fo": int(fo), "_fe": int(fe),
    })
    event_dfs[ev_num] = ev
    print(f"  E{ev_num}: {pf:.0f}N  {ev.loc[lpf,'resultant_velocity']:.2f}m/s  acc_max={ev['resultant_acceleration'].max():.0f}m/s²")

# ── Plotly helpers ────────────────────────────────────────────────────────────
BASE_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Inter, sans-serif", size=12),
    paper_bgcolor="white", plot_bgcolor="white",
    hoverlabel=dict(font_size=12, font_family="Inter"),
)

def jfig(fig): return pio.to_json(fig)

# ── 1. FULL SIGNAL ────────────────────────────────────────────────────────────
print("Signal chart...")
fig_sig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07,
    subplot_titles=["<b>Resultant Force (N)</b>", "<b>Resultant Acceleration (m/s²) — includes gravity baseline ~10 m/s²</b>"])
fig_sig.add_trace(go.Scatter(x=raw["Time"], y=raw["resultant_force"],
    line=dict(color="#457B9D", width=1), fill="tozeroy",
    fillcolor="rgba(69,123,157,0.12)", name="Force"), row=1, col=1)
fig_sig.add_trace(go.Scatter(x=raw["Time"], y=raw["resultant_acceleration"],
    line=dict(color="#F4A261", width=1), fill="tozeroy",
    fillcolor="rgba(244,162,97,0.12)", name="Acceleration"), row=2, col=1)
for i, (pi, res) in enumerate(zip(peak_indices, all_results)):
    c  = EVENT_COLORS[i]; pt = raw.loc[pi, "Time"]; pf_val = raw.loc[pi, "resultant_force"]
    fig_sig.add_vrect(x0=pt-WINDOW_SIZE/2, x1=pt+WINDOW_SIZE/2,
        fillcolor=c, opacity=0.13, layer="below", line_width=0, row=1, col=1)
    fig_sig.add_trace(go.Scatter(x=[pt], y=[pf_val], mode="markers+text",
        marker=dict(symbol="x", size=14, color=c, line=dict(width=2.5)),
        text=[f"E{i+1}<br>{pf_val:.0f}N"], textposition="top center",
        textfont=dict(size=10, color=c), name=f"Strike {i+1}", showlegend=True), row=1, col=1)
fig_sig.update_layout(height=600, **BASE_LAYOUT,
    margin=dict(l=60, r=20, t=50, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
fig_sig.update_xaxes(title_text="Time (s)", row=2, col=1, showgrid=True, gridcolor="#f0f0f0")
fig_sig.update_yaxes(title_text="N",    row=1, col=1, showgrid=True, gridcolor="#f0f0f0")
fig_sig.update_yaxes(title_text="m/s²", row=2, col=1, showgrid=True, gridcolor="#f0f0f0")
json_sig = jfig(fig_sig)

# ── 2. PER-EVENT ──────────────────────────────────────────────────────────────
print("Per-event charts...")
json_events = []
for ev_num, (res, pi) in enumerate(zip(all_results, peak_indices), 1):
    ev  = event_dfs[ev_num]
    c   = EVENT_COLORS[ev_num-1]
    tms = ev["Time_rel"] * 1000
    lpf = res["_lpf"]; fo = res["_fo"]; fe = res["_fe"]; ao = res["_ao"]

    fig_ev = make_subplots(rows=2, cols=2, vertical_spacing=0.18, horizontal_spacing=0.12,
        subplot_titles=["<b>Force (N)</b>","<b>Acceleration (m/s²)</b>",
                        "<b>Fist Velocity (m/s)</b>","<b>Velocity by Axis (m/s)</b>"])

    # Force
    fig_ev.add_trace(go.Scatter(x=tms, y=ev["resultant_force"],
        fill="tozeroy", fillcolor="rgba(69,123,157,0.15)",
        line=dict(color="#457B9D", width=2.5), name="Force", showlegend=False), row=1, col=1)
    for xv, lbl, col_ in [(tms.iloc[fo],"Onset","#2A9D8F"),
                           (tms.iloc[lpf],"Peak","#C62828"),
                           (tms.iloc[fe],"End","#555")]:
        fig_ev.add_vline(x=float(xv), line_dash="dash", line_color=col_, line_width=1.8,
            annotation_text=lbl, annotation_font_size=9, annotation_font_color=col_,
            row=1, col=1)

    # Acceleration
    fig_ev.add_trace(go.Scatter(x=tms, y=ev["resultant_acceleration"],
        fill="tozeroy", fillcolor="rgba(244,162,97,0.15)",
        line=dict(color="#F4A261", width=2.5), name="Acc", showlegend=False), row=1, col=2)
    fig_ev.add_vline(x=float(tms.iloc[ao]),  line_dash="dash", line_color="#9B5DE5",
        line_width=1.8, annotation_text="Motion onset", annotation_font_size=9,
        annotation_font_color="#9B5DE5", row=1, col=2)
    fig_ev.add_vline(x=float(tms.iloc[lpf]), line_dash="dash", line_color="#C62828",
        line_width=1.8, annotation_text="Peak F", annotation_font_size=9,
        annotation_font_color="#C62828", row=1, col=2)

    # Velocity resultant
    fig_ev.add_trace(go.Scatter(x=tms, y=ev["resultant_velocity"],
        line=dict(color="#9B5DE5", width=3), name="Vel", showlegend=False), row=2, col=1)
    fig_ev.add_vline(x=float(tms.iloc[lpf]), line_dash="dash", line_color="#C62828",
        line_width=2, row=2, col=1)
    fig_ev.add_trace(go.Scatter(
        x=[float(tms.iloc[lpf])], y=[res["vel_at_peak"]],
        mode="markers+text", marker=dict(color="#C62828", size=12, symbol="circle"),
        text=[f"  {res['vel_at_peak']:.2f} m/s"], textposition="middle right",
        textfont=dict(size=12, color="#C62828", family="Inter"), showlegend=False), row=2, col=1)

    # Velocity XYZ
    for ax_lbl, col_name, col_c in [("X","vel_x","#E63946"),("Y","vel_y","#2A9D8F"),("Z","vel_z","#457B9D")]:
        fig_ev.add_trace(go.Scatter(x=tms, y=ev[col_name],
            line=dict(color=col_c, width=2), name=ax_lbl, showlegend=True), row=2, col=2)
    fig_ev.add_vline(x=float(tms.iloc[lpf]), line_dash="dash", line_color="#C62828",
        line_width=1.5, opacity=0.5, row=2, col=2)

    fig_ev.update_layout(height=580, **BASE_LAYOUT,
        margin=dict(l=60, r=20, t=55, b=50),
        title=dict(text=f"<b>Strike {ev_num}</b>  ·  t={res['peak_time']}s  ·  Peak: {res['peak_force']:.0f} N  ·  Vel. Impact: {res['vel_at_peak']:.2f} m/s",
                   font=dict(size=14), x=0),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="right", x=1))
    for r_, c_, u_ in [(1,1,"N"),(1,2,"m/s²"),(2,1,"m/s"),(2,2,"m/s")]:
        fig_ev.update_yaxes(title_text=u_, row=r_, col=c_, showgrid=True, gridcolor="#f0f0f0")
        fig_ev.update_xaxes(title_text="Time (ms)", row=r_, col=c_, showgrid=True, gridcolor="#f0f0f0")
    json_events.append(jfig(fig_ev))

# ── 3. RFD per event (single chart) ──────────────────────────────────────────
print("RFD chart...")
fig_rfd = go.Figure()
for i, (res, ev_num) in enumerate(zip(all_results, range(1, NUM_EVENTS+1))):
    ev  = event_dfs[ev_num]
    tms = ev["Time_rel"] * 1000
    rfd_clean = ev["RFD"].dropna()
    fig_rfd.add_trace(go.Scatter(x=tms.iloc[rfd_clean.index], y=rfd_clean,
        line=dict(color=EVENT_COLORS[i], width=1.8),
        name=f"Strike {ev_num} ({res['peak_force']:.0f}N)"))
fig_rfd.update_layout(height=380, **BASE_LAYOUT,
    margin=dict(l=60, r=20, t=40, b=40),
    xaxis_title="Time (ms)", yaxis_title="RFD (N/s)",
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1))
json_rfd = jfig(fig_rfd)

# ── 4. COMPARISON ─────────────────────────────────────────────────────────────
print("Comparison chart...")
compare_metrics = [
    ("Peak Force",       "peak_force",      "N",    0),
    ("Impact Velocity",  "vel_at_peak",     "m/s",  0),
    ("Max RFD",          "rfd_max",         "N/s",  1),
    ("Total Impulse",    "impulse_total",   "N·s",  1),
    ("Time to Peak",     "time_to_peak_ms", "ms",   2),
    ("Contact Duration", "contact_dur_ms",  "ms",   2),
]
fig_cmp = make_subplots(rows=3, cols=2, vertical_spacing=0.12, horizontal_spacing=0.1,
    subplot_titles=[f"<b>{m[0]}</b> ({m[2]})" for m in compare_metrics])
ev_labels = [f"E{r['ev']}" for r in all_results]
for idx, (label, key, unit, _) in enumerate(compare_metrics):
    r_, c_ = divmod(idx, 2)
    vals   = [r[key] for r in all_results]
    mean_v = np.mean(vals)
    max_i  = int(np.argmax(vals))
    bar_colors = [EVENT_COLORS[i] for i in range(len(vals))]
    fig_cmp.add_trace(go.Bar(
        x=ev_labels, y=vals, marker_color=bar_colors,
        text=[f"<b>{v:.1f}</b>" for v in vals],
        textposition="outside", textfont=dict(size=11),
        showlegend=False, name=label,
        hovertemplate=f"<b>%{{x}}</b><br>{label}: %{{y:.1f}} {unit}<extra></extra>"),
        row=r_+1, col=c_+1)
    fig_cmp.add_hline(y=mean_v, line_dash="dot", line_color="#aaa", line_width=1.5,
        annotation_text=f"μ={mean_v:.1f}", annotation_font_size=9,
        annotation_position="top right", row=r_+1, col=c_+1)
    fig_cmp.update_yaxes(title_text=unit, row=r_+1, col=c_+1, showgrid=True, gridcolor="#f0f0f0")
    fig_cmp.update_xaxes(showgrid=False, row=r_+1, col=c_+1)
fig_cmp.update_layout(height=780, **BASE_LAYOUT, margin=dict(l=60, r=20, t=50, b=40))
json_cmp = jfig(fig_cmp)

# ── 5. SUMMARY DATA for table + cards ─────────────────────────────────────────
ALL_METRICS = [
    ("Peak Force",              "peak_force",      "N",    True,  1),
    ("Net Peak Force",          "net_peak_force",  "N",    True,  1),
    ("Time to Peak",            "time_to_peak_ms", "ms",   False, 1),
    ("Contact Duration",        "contact_dur_ms",  "ms",   False, 1),
    ("Kinematic Strike Dur.",   "kin_dur_ms",       "ms",   False, 1),
    ("Total Impulse",           "impulse_total",   "N·s",  True,  2),
    ("Impulse to Peak",         "impulse_to_peak", "N·s",  True,  2),
    ("Max RFD",                 "rfd_max",         "N/s",  True,  0),
    ("Max Jerk",                "jerk_max",        "N/s²", True,  0),
    ("Force @ 10 ms",           "f_10ms",          "N",    True,  1),
    ("RFD 0–10 ms",             "rfd_0_10",        "N/s",  True,  0),
    ("Force @ 20 ms",           "f_20ms",          "N",    True,  1),
    ("RFD 0–20 ms",             "rfd_0_20",        "N/s",  True,  0),
    ("Impact Velocity",         "vel_at_peak",     "m/s",  True,  2),
    ("Max Velocity",            "vel_max",         "m/s",  True,  2),
    ("Acceleration at Impact",  "acc_at_peak",     "m/s²", True,  1),
    ("Max Acceleration",        "acc_max",         "m/s²", True,  1),
]

# ── Build HTML components ─────────────────────────────────────────────────────
avg_force = np.mean([r["peak_force"]  for r in all_results])
avg_vel   = np.mean([r["vel_at_peak"] for r in all_results])
avg_rfd   = np.mean([r["rfd_max"]     for r in all_results])
avg_imp   = np.mean([r["impulse_total"] for r in all_results])

# Strike selector buttons
ev_btns_html = ""
ev_panes_html = ""
for i, r in enumerate(all_results):
    active = "active" if i == 0 else ""
    c = EVENT_COLORS[i]
    ev_btns_html += f'<button class="tab-btn {active}" onclick="showEvent({i})" id="evbtn-{i}" style="--ec:{c}">Strike {r["ev"]}<br><small>{r["peak_force"]:.0f} N &nbsp;·&nbsp; {r["vel_at_peak"]:.2f} m/s</small></button>\n'
    # per-event metrics cards row
    metric_cards = ""
    for lbl, key, unit, higher_better, dec in ALL_METRICS[:8]:
        val = r[key]
        fmt = f"{val:.{dec}f}"
        metric_cards += f"""<div class="metric-mini"><span class="metric-mini-lbl">{lbl}</span><span class="metric-mini-val">{fmt}</span><span class="metric-mini-unit">{unit}</span></div>\n"""
    metric_cards2 = ""
    for lbl, key, unit, higher_better, dec in ALL_METRICS[8:]:
        val = r[key]
        fmt = f"{val:.{dec}f}"
        metric_cards2 += f"""<div class="metric-mini"><span class="metric-mini-lbl">{lbl}</span><span class="metric-mini-val">{fmt}</span><span class="metric-mini-unit">{unit}</span></div>\n"""

    ev_panes_html += f"""
    <div class="event-pane {'active' if i==0 else ''}" id="evpane-{i}">
      <div id="ev-plot-{i}" class="chart-box"></div>
      <details open class="metrics-details">
        <summary>All metrics — Strike {r["ev"]}</summary>
        <div class="metric-mini-grid">{metric_cards}{metric_cards2}</div>
      </details>
    </div>"""

# Full summary table (color coded best/worst per row)
def color_td(val, all_vals, higher_better, dec):
    fmt = f"{val:.{dec}f}"
    if len(all_vals) < 2:
        return f"<td>{fmt}</td>"
    best  = max(all_vals) if higher_better else min(all_vals)
    worst = min(all_vals) if higher_better else max(all_vals)
    if val == best:
        return f'<td class="best">{fmt}</td>'
    elif val == worst:
        return f'<td class="worst">{fmt}</td>'
    return f"<td>{fmt}</td>"

summary_thead_html = "<tr><th>Metric</th><th>Unit</th>" + "".join(f"<th>E{r['ev']}</th>" for r in all_results) + "<th>Mean</th><th>SD</th></tr>"
summary_tbody_html = ""
for lbl, key, unit, higher_better, dec in ALL_METRICS:
    vals = [r[key] for r in all_results]
    mean_v = np.mean(vals); sd_v = np.std(vals)
    cells = "".join(color_td(v, vals, higher_better, dec) for v in vals)
    summary_tbody_html += f"<tr><td class='metric-name'>{lbl}</td><td class='unit-col'>{unit}</td>{cells}<td class='mean-col'>{mean_v:.{dec}f}</td><td class='sd-col'>±{sd_v:.{dec}f}</td></tr>\n"

# Strike overview cards
strike_cards_html = ""
for i, r in enumerate(all_results):
    c = EVENT_COLORS[i]
    strike_cards_html += f"""
    <div class="strike-card" style="border-top:4px solid {c}">
      <div class="sc-header" style="color:{c}">Strike {r['ev']} <span>t = {r['peak_time']} s</span></div>
      <div class="sc-main">{r['peak_force']:.0f} <small>N</small></div>
      <div class="sc-sub">{r['vel_at_peak']:.2f} m/s &nbsp;|&nbsp; {r['rfd_max']/1000:.0f}k N/s</div>
      <div class="sc-row"><span>Time→Peak</span><b>{r['time_to_peak_ms']:.0f} ms</b></div>
      <div class="sc-row"><span>Contact</span><b>{r['contact_dur_ms']:.0f} ms</b></div>
      <div class="sc-row"><span>Impulse</span><b>{r['impulse_total']:.2f} N·s</b></div>
      <div class="sc-row"><span>Acc@Peak</span><b>{r['acc_at_peak']:.0f} m/s²</b></div>
    </div>"""

print("Assembling HTML...")
HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Boxing Biomechanics Lab — {ATHLETE}</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800;900&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
:root{{--red:#C62828;--gold:#F9A825;--dark:#1A1A2E;--mid:#16213E;--bg:#F4F6FA;--card:#fff}}
body{{font-family:'Inter',sans-serif;background:var(--bg);color:var(--dark);line-height:1.5}}

/* ── NAV ── */
nav{{background:linear-gradient(135deg,var(--dark) 0%,var(--red) 100%);padding:.9rem 2rem;
  display:flex;align-items:center;justify-content:space-between;
  position:sticky;top:0;z-index:100;box-shadow:0 2px 16px rgba(0,0,0,.35)}}
.brand{{color:#fff;font-size:1.2rem;font-weight:900;letter-spacing:-.5px}}
.brand span{{color:var(--gold)}}
.nav-links{{display:flex;gap:.4rem;flex-wrap:wrap}}
.nav-links a{{color:rgba(255,255,255,.7);text-decoration:none;font-size:.82rem;font-weight:600;
  padding:.35rem .85rem;border-radius:20px;transition:all .2s;white-space:nowrap}}
.nav-links a:hover,.nav-links a.active{{background:rgba(255,255,255,.18);color:#fff}}

/* ── HERO ── */
.hero{{background:linear-gradient(135deg,var(--dark) 0%,#7B0000 100%);
  padding:2.5rem 2rem 2rem;text-align:center;color:#fff}}
.hero h1{{font-size:2.4rem;font-weight:900;margin-bottom:.4rem}}
.hero p{{font-size:1rem;opacity:.82}}
.badges{{display:flex;gap:.5rem;justify-content:center;flex-wrap:wrap;margin-top:.9rem}}
.badge{{background:rgba(249,168,37,.18);color:var(--gold);border:1px solid var(--gold);
  border-radius:20px;padding:.25rem .9rem;font-size:.75rem;font-weight:700;letter-spacing:.8px}}

/* ── MAIN ── */
main{{max-width:1300px;margin:0 auto;padding:1.8rem 1.5rem}}
.page{{display:none}}.page.active{{display:block}}

/* ── SECTION ── */
.sec-title{{font-size:1.1rem;font-weight:800;color:var(--dark);
  border-left:4px solid var(--red);padding-left:.7rem;margin:1.5rem 0 1rem}}

/* ── AVG ROW ── */
.avg-row{{display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin:1.5rem 0 1.5rem}}
.avg-card{{background:var(--card);border-radius:12px;padding:1.1rem 1rem;
  box-shadow:0 2px 10px rgba(0,0,0,.06);border-top:3px solid var(--red);text-align:center}}
.avg-card .lbl{{font-size:.68rem;text-transform:uppercase;letter-spacing:.5px;color:#888;font-weight:700}}
.avg-card .val{{font-size:1.9rem;font-weight:900;color:var(--red);margin:.2rem 0}}
.avg-card .unt{{font-size:.78rem;color:#aaa}}

/* ── STRIKE CARDS ── */
.strike-cards{{display:grid;grid-template-columns:repeat(5,1fr);gap:.9rem;margin-bottom:1.8rem}}
.strike-card{{background:var(--card);border-radius:12px;padding:1rem;
  box-shadow:0 2px 10px rgba(0,0,0,.06)}}
.sc-header{{font-size:.78rem;font-weight:800;text-transform:uppercase;letter-spacing:.5px;
  display:flex;justify-content:space-between;align-items:center;margin-bottom:.5rem}}
.sc-header span{{font-size:.7rem;opacity:.6;font-weight:400}}
.sc-main{{font-size:2rem;font-weight:900;color:var(--dark);line-height:1}}
.sc-main small{{font-size:.85rem;font-weight:400;color:#999}}
.sc-sub{{font-size:.78rem;color:#666;margin:.2rem 0 .6rem}}
.sc-row{{display:flex;justify-content:space-between;font-size:.78rem;
  padding:.2rem 0;border-bottom:1px solid #f5f5f5}}
.sc-row span{{color:#888}}.sc-row b{{color:var(--dark)}}

/* ── CHART BOX ── */
.chart-box{{background:var(--card);border-radius:14px;padding:1rem;
  box-shadow:0 2px 12px rgba(0,0,0,.07);margin-bottom:1rem}}

/* ── EVENT TABS ── */
.ev-btns{{display:flex;gap:.5rem;flex-wrap:wrap;margin-bottom:1rem}}
.tab-btn{{background:var(--card);border:2px solid #ddd;border-radius:10px;
  padding:.5rem 1.1rem;cursor:pointer;font-family:'Inter',sans-serif;
  font-size:.82rem;font-weight:700;color:#666;transition:all .2s;text-align:center;line-height:1.3}}
.tab-btn:hover{{border-color:var(--ec);color:var(--ec)}}
.tab-btn.active{{background:var(--ec);border-color:var(--ec);color:#fff}}
.event-pane{{display:none}}.event-pane.active{{display:block}}

/* ── METRIC MINI GRID ── */
.metrics-details{{margin-top:.5rem}}
.metrics-details summary{{cursor:pointer;font-weight:700;font-size:.9rem;
  color:var(--red);padding:.5rem 0;user-select:none}}
.metric-mini-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:.6rem;padding:.8rem 0}}
.metric-mini{{background:var(--bg);border-radius:8px;padding:.6rem .8rem;
  display:flex;flex-direction:column;border-left:3px solid var(--red)}}
.metric-mini-lbl{{font-size:.67rem;text-transform:uppercase;letter-spacing:.4px;color:#888;font-weight:600}}
.metric-mini-val{{font-size:1.3rem;font-weight:800;color:var(--dark);line-height:1.2}}
.metric-mini-unit{{font-size:.7rem;color:#aaa}}

/* ── SUMMARY TABLE ── */
.tbl-wrap{{background:var(--card);border-radius:14px;overflow-x:auto;
  box-shadow:0 2px 12px rgba(0,0,0,.07)}}
.sum-table{{width:100%;border-collapse:collapse;font-size:.83rem;white-space:nowrap}}
.sum-table thead{{background:var(--dark);color:#fff;position:sticky;top:0}}
.sum-table th{{padding:.65rem 1rem;text-align:center;font-weight:700;font-size:.78rem}}
.sum-table th:first-child{{text-align:left}}
.sum-table td{{padding:.6rem .9rem;border-bottom:1px solid #f0f0f0;text-align:center}}
.sum-table tr:hover td{{background:#fef9f9}}
.sum-table .metric-name{{text-align:left;font-weight:600;color:var(--dark);white-space:normal;min-width:160px}}
.sum-table .unit-col{{color:#999;font-size:.75rem}}
.sum-table .mean-col{{font-weight:700;color:#457B9D}}
.sum-table .sd-col{{color:#aaa;font-size:.78rem}}
.sum-table td.best{{background:#e8f8f0;color:#1a7f4b;font-weight:800}}
.sum-table td.worst{{background:#fef0f0;color:#c62828;font-weight:700}}

/* ── FOOTER ── */
footer{{text-align:center;padding:2rem;color:#bbb;font-size:.78rem;border-top:1px solid #eee;margin-top:2rem}}

@media(max-width:900px){{
  .avg-row{{grid-template-columns:repeat(2,1fr)}}
  .strike-cards{{grid-template-columns:repeat(2,1fr)}}
  nav .brand{{font-size:1rem}}
  .hero h1{{font-size:1.7rem}}
}}
</style>
</head>
<body>

<nav>
  <div class="brand">🥊 Boxing <span>Biomechanics</span> Lab</div>
  <div class="nav-links">
    <a class="active" href="#" onclick="showPage('signal',this);return false">Signal</a>
    <a href="#" onclick="showPage('events',this);return false">Per Event</a>
    <a href="#" onclick="showPage('rfd',this);return false">RFD</a>
    <a href="#" onclick="showPage('compare',this);return false">Comparison</a>
    <a href="#" onclick="showPage('summary',this);return false">Summary</a>
  </div>
</nav>

<div class="hero">
  <h1>🥊 Boxing Biomechanics Lab</h1>
  <p>Kinematic &amp; kinetic strike analysis &nbsp;·&nbsp; Noraxon IMU + Force Sensor</p>
  <div class="badges">
    <span class="badge">ATHLETE: {ATHLETE}</span>
    <span class="badge">LEFT HAND PAD</span>
    <span class="badge">{NUM_EVENTS} STRIKES</span>
    <span class="badge">1000 Hz</span>
  </div>
</div>

<main>

<!-- ── GLOBAL KPIs ── -->
<div class="avg-row">
  <div class="avg-card"><div class="lbl">Strikes Analyzed</div><div class="val">{NUM_EVENTS}</div><div class="unt">events</div></div>
  <div class="avg-card"><div class="lbl">Mean Peak Force</div><div class="val">{avg_force:.0f}</div><div class="unt">N</div></div>
  <div class="avg-card"><div class="lbl">Mean Impact Velocity</div><div class="val">{avg_vel:.2f}</div><div class="unt">m/s</div></div>
  <div class="avg-card"><div class="lbl">Mean Max RFD</div><div class="val">{avg_rfd/1000:.0f}k</div><div class="unt">N/s</div></div>
</div>

<!-- ── STRIKE CARDS ── -->
<div class="strike-cards">{strike_cards_html}</div>

<!-- ═══ PAGE: SIGNAL ═══ -->
<div class="page active" id="page-signal">
  <div class="sec-title">Full Recording — Resultant Force &amp; Acceleration</div>
  <p style="font-size:.82rem;color:#888;margin-bottom:.8rem">
    ⚠️ Acceleration includes the gravity baseline (~10 m/s²).
    Peak values represent total IMU acceleration (gravitational + linear).
    Peak punch acceleration: {max(r['acc_max'] for r in all_results):.0f} m/s² = {max(r['acc_max'] for r in all_results)/9.81:.0f} g
  </p>
  <div class="chart-box" id="plot-signal"></div>
</div>

<!-- ═══ PAGE: EVENTS ═══ -->
<div class="page" id="page-events">
  <div class="sec-title">Per-Event Analysis</div>
  <div class="ev-btns">{ev_btns_html}</div>
  {ev_panes_html}
</div>

<!-- ═══ PAGE: RFD ═══ -->
<div class="page" id="page-rfd">
  <div class="sec-title">Rate of Force Development (RFD) — All Strikes Overlaid</div>
  <div class="chart-box" id="plot-rfd"></div>
</div>

<!-- ═══ PAGE: COMPARE ═══ -->
<div class="page" id="page-compare">
  <div class="sec-title">Strike Comparison — Key Metrics</div>
  <div class="chart-box" id="plot-compare"></div>
</div>

<!-- ═══ PAGE: SUMMARY ═══ -->
<div class="page" id="page-summary">
  <div class="sec-title">Complete Metrics Summary</div>
  <p style="font-size:.82rem;color:#888;margin-bottom:.8rem">
    <span style="background:#e8f8f0;color:#1a7f4b;padding:.1rem .4rem;border-radius:4px;font-weight:700">Best</span> &nbsp;
    <span style="background:#fef0f0;color:#c62828;padding:.1rem .4rem;border-radius:4px;font-weight:700">Worst</span>
    &nbsp; per metric across the {NUM_EVENTS} strikes.
  </p>
  <div class="tbl-wrap">
    <table class="sum-table">
      <thead>{summary_thead_html}</thead>
      <tbody>{summary_tbody_html}</tbody>
    </table>
  </div>
</div>

</main>
<footer>Boxing Biomechanics Lab &nbsp;·&nbsp; {ATHLETE} &nbsp;·&nbsp; Noraxon Data &nbsp;·&nbsp; 1000 Hz</footer>

<script>
const SIG  = {json_sig};
const RFD  = {json_rfd};
const CMP  = {json_cmp};
const EVS  = [{','.join(json_events)}];

let rendered = {{}};

function showPage(name, el) {{
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-links a').forEach(a => a.classList.remove('active'));
  document.getElementById('page-' + name).classList.add('active');
  if (el) el.classList.add('active');
  if (!rendered[name]) {{
    if (name === 'signal')  Plotly.react('plot-signal',  SIG.data, SIG.layout, {{responsive:true}});
    if (name === 'rfd')     Plotly.react('plot-rfd',     RFD.data, RFD.layout, {{responsive:true}});
    if (name === 'compare') Plotly.react('plot-compare', CMP.data, CMP.layout, {{responsive:true}});
    rendered[name] = true;
  }}
}}

let curEv = 0;
let evRendered = {{}};
function showEvent(i) {{
  document.querySelectorAll('.event-pane').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('evpane-' + i).classList.add('active');
  document.getElementById('evbtn-'  + i).classList.add('active');
  curEv = i;
  if (!evRendered[i]) {{
    const d = EVS[i];
    Plotly.react('ev-plot-' + i, d.data, d.layout, {{responsive:true}});
    evRendered[i] = true;
  }}
}}

window.addEventListener('load', () => {{
  Plotly.react('plot-signal', SIG.data, SIG.layout, {{responsive:true}});
  rendered['signal'] = true;
  showEvent(0);
}});
</script>
</body>
</html>"""

os.makedirs("docs", exist_ok=True)
for path in ["docs/index.html", "index.html"]:
    with open(path, "w", encoding="utf-8") as f:
        f.write(HTML)
    print(f"Written: {path}  ({os.path.getsize(path)//1024} KB)")

print("\nDone.")
