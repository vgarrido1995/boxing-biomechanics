import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

MILLI_G_TO_MPS2 = 9.80665 / 1000
FORCE_THRESHOLD = 300
MIN_TIME_SEP = 0.5
WINDOW_SIZE = 0.4
ACC_THRESHOLD = 12
ONSET_THRESHOLD = 20
NUM_EVENTS = 5
FILENAME = 'K001_J left hand pad.xlsx'

raw = pd.read_excel(FILENAME)
raw[['1x', '1y', '1z']] = raw[['1x', '1y', '1z']] * MILLI_G_TO_MPS2
raw['resultant_acceleration'] = np.sqrt(raw['1x']**2 + raw['1y']**2 + raw['1z']**2)
raw['resultant_force'] = np.sqrt(raw['fx']**2 + raw['fy']**2 + raw['fz']**2)

def find_top_peaks(data):
    valid = data[data['resultant_force'] >= FORCE_THRESHOLD]
    sorted_data = valid.sort_values('resultant_force', ascending=False)
    peaks = []
    for idx, row in sorted_data.iterrows():
        if len(peaks) >= NUM_EVENTS:
            break
        diffs = [abs(row['Time'] - data.loc[p, 'Time']) for p in peaks]
        if all(d >= MIN_TIME_SEP for d in diffs):
            peaks.append(idx)
    peaks.sort(key=lambda i: data.loc[i, 'Time'])
    return peaks

def integrate_velocity(acc_values, times):
    vel = np.zeros(len(acc_values))
    for i in range(1, len(vel)):
        dt = times[i] - times[i-1]
        vel[i] = vel[i-1] + acc_values[i-1] * dt
    return vel

def find_strike_onset(force_series, peak_idx_local, threshold=ONSET_THRESHOLD):
    for i in range(peak_idx_local, -1, -1):
        if force_series.iloc[i] < threshold:
            return i
    return 0

def find_strike_end(force_series, peak_idx_local, threshold=ONSET_THRESHOLD):
    for i in range(peak_idx_local, len(force_series)):
        if force_series.iloc[i] < threshold:
            return i
    return len(force_series) - 1

def find_acc_onset(acc_series, threshold=ACC_THRESHOLD):
    for i, val in enumerate(acc_series):
        if val > threshold:
            return i
    return 0

peak_indices = find_top_peaks(raw)
all_results = []

for ev_num, peak_idx in enumerate(peak_indices, 1):
    peak_time = raw.loc[peak_idx, 'Time']
    t_start = peak_time - WINDOW_SIZE / 2
    t_end   = peak_time + WINDOW_SIZE / 2
    ev = raw[(raw['Time'] >= t_start) & (raw['Time'] <= t_end)].copy()
    ev.reset_index(drop=True, inplace=True)
    ev['Time_rel'] = ev['Time'] - t_start

    times = ev['Time'].values
    ev['vel_x'] = integrate_velocity(ev['1x'].values, times)
    ev['vel_y'] = integrate_velocity(ev['1y'].values, times)
    ev['vel_z'] = integrate_velocity(ev['1z'].values, times)
    ev['resultant_velocity'] = np.sqrt(ev['vel_x']**2 + ev['vel_y']**2 + ev['vel_z']**2)

    local_peak_f_idx = ev['resultant_force'].idxmax()
    acc_onset_idx    = find_acc_onset(ev['resultant_acceleration'])
    force_onset_idx  = find_strike_onset(ev['resultant_force'], local_peak_f_idx)
    force_end_idx    = find_strike_end(ev['resultant_force'], local_peak_f_idx)

    peak_force      = ev.loc[local_peak_f_idx, 'resultant_force']
    force_onset_val = ev.loc[force_onset_idx, 'resultant_force']
    net_peak_force  = peak_force - force_onset_val
    t_force_onset   = ev.loc[force_onset_idx, 'Time']
    t_force_end     = ev.loc[force_end_idx, 'Time']
    t_peak_force    = ev.loc[local_peak_f_idx, 'Time']
    t_acc_onset     = ev.loc[acc_onset_idx, 'Time']
    time_to_peak_ms = (t_peak_force - t_force_onset) * 1000
    contact_dur_ms  = (t_force_end - t_force_onset) * 1000
    strike_dur_kin  = (t_peak_force - t_acc_onset)

    contact_data    = ev.loc[force_onset_idx:force_end_idx]
    impulse_total   = np.trapz(contact_data['resultant_force'], x=contact_data['Time'])
    impulse_to_peak = np.trapz(
        ev.loc[force_onset_idx:local_peak_f_idx, 'resultant_force'],
        x=ev.loc[force_onset_idx:local_peak_f_idx, 'Time']
    )

    dt_mean = ev['Time'].diff().mean()
    ev['RFD']  = ev['resultant_force'].diff() / dt_mean
    ev['Jerk'] = ev['RFD'].diff() / dt_mean
    max_rfd  = ev['RFD'].max()

    def force_at_ms(ms):
        target = t_force_onset + ms / 1000.0
        idx = (ev['Time'] - target).abs().idxmin()
        return ev.loc[idx, 'resultant_force']

    f_10ms   = force_at_ms(10)
    f_20ms   = force_at_ms(20)
    rfd_0_10 = (f_10ms - force_onset_val) / 0.010
    rfd_0_20 = (f_20ms - force_onset_val) / 0.020

    velocity_at_peak = ev.loc[local_peak_f_idx, 'resultant_velocity']
    max_velocity     = ev['resultant_velocity'].max()
    acc_at_peak      = ev.loc[local_peak_f_idx, 'resultant_acceleration']

    result = {
        'Evento': ev_num,
        'Fuerza Pico (N)': round(peak_force, 2),
        'Fuerza Neta Pico (N)': round(net_peak_force, 2),
        'Tiempo hasta Pico (ms)': round(time_to_peak_ms, 1),
        'Duracion Contacto (ms)': round(contact_dur_ms, 1),
        'Impulso Total (Ns)': round(impulse_total, 3),
        'Impulso hasta Pico (Ns)': round(impulse_to_peak, 3),
        'RFD Max (N/s)': round(max_rfd, 0),
        'Fuerza 10ms (N)': round(f_10ms, 2),
        'RFD 0-10ms (N/s)': round(rfd_0_10, 0),
        'Fuerza 20ms (N)': round(f_20ms, 2),
        'RFD 0-20ms (N/s)': round(rfd_0_20, 0),
        'Velocidad en Pico (m/s)': round(velocity_at_peak, 3),
        'Velocidad Max (m/s)': round(max_velocity, 3),
        'Aceleracion en Pico (m/s2)': round(acc_at_peak, 1),
        'Duracion Golpe Kin (ms)': round(strike_dur_kin * 1000, 1),
    }
    all_results.append(result)

    # ── PLOT ──────────────────────────────────────────────────────────
    t_rel = ev['Time_rel'] * 1000  # ms
    fig = plt.figure(figsize=(16, 11))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.32)
    ax_force   = fig.add_subplot(gs[0, 0])
    ax_acc     = fig.add_subplot(gs[0, 1])
    ax_vel     = fig.add_subplot(gs[1, 0])
    ax_vel_xyz = fig.add_subplot(gs[1, 1])
    ax_rfd     = fig.add_subplot(gs[2, 0])
    ax_table   = fig.add_subplot(gs[2, 1])

    title = f'Evento {ev_num}  |  t={peak_time:.3f}s  |  Fuerza Pico: {peak_force:.0f} N  |  Vel. Impacto: {velocity_at_peak:.2f} m/s'
    fig.suptitle(title, fontsize=13, fontweight='bold')

    # Fuerza
    ax_force.plot(t_rel, ev['resultant_force'], color='steelblue', linewidth=2)
    ax_force.fill_between(t_rel, ev['resultant_force'], alpha=0.15, color='steelblue')
    ax_force.axvline(t_rel.iloc[force_onset_idx], color='green', ls='--', lw=1.5, label='Inicio')
    ax_force.axvline(t_rel.iloc[local_peak_f_idx], color='red', ls='--', lw=1.5, label='Pico')
    ax_force.axvline(t_rel.iloc[force_end_idx], color='black', ls='--', lw=1.5, label='Fin')
    ax_force.set_title('Fuerza Resultante'); ax_force.set_ylabel('N'); ax_force.set_xlabel('ms')
    ax_force.legend(fontsize=8); ax_force.grid(True, alpha=0.3)

    # Aceleracion
    ax_acc.plot(t_rel, ev['resultant_acceleration'], color='darkorange', linewidth=2)
    ax_acc.axvline(t_rel.iloc[acc_onset_idx], color='purple', ls='--', lw=1.5, label='Inicio golpe')
    ax_acc.axvline(t_rel.iloc[local_peak_f_idx], color='red', ls='--', lw=1.5, label='Pico fuerza')
    ax_acc.set_title('Aceleracion Resultante'); ax_acc.set_ylabel('m/s2'); ax_acc.set_xlabel('ms')
    ax_acc.legend(fontsize=8); ax_acc.grid(True, alpha=0.3)

    # Velocidad resultante
    ax_vel.plot(t_rel, ev['resultant_velocity'], color='purple', linewidth=2)
    ax_vel.axvline(t_rel.iloc[local_peak_f_idx], color='red', ls='--', lw=1.5)
    ax_vel.scatter(t_rel.iloc[local_peak_f_idx], velocity_at_peak, color='red', s=80, zorder=5)
    ax_vel.annotate(f'{velocity_at_peak:.2f} m/s',
                    (t_rel.iloc[local_peak_f_idx], velocity_at_peak),
                    xytext=(8, -12), textcoords='offset points', fontsize=9, color='red', fontweight='bold')
    ax_vel.set_title('Velocidad Resultante del Puno'); ax_vel.set_ylabel('m/s'); ax_vel.set_xlabel('ms')
    ax_vel.grid(True, alpha=0.3)

    # Velocidad XYZ
    ax_vel_xyz.plot(t_rel, ev['vel_x'], label='X', color='red',   linewidth=1.5)
    ax_vel_xyz.plot(t_rel, ev['vel_y'], label='Y', color='green', linewidth=1.5)
    ax_vel_xyz.plot(t_rel, ev['vel_z'], label='Z', color='blue',  linewidth=1.5)
    ax_vel_xyz.axvline(t_rel.iloc[local_peak_f_idx], color='red', ls='--', lw=1, alpha=0.5)
    ax_vel_xyz.set_title('Velocidad por Ejes'); ax_vel_xyz.set_ylabel('m/s'); ax_vel_xyz.set_xlabel('ms')
    ax_vel_xyz.legend(fontsize=8); ax_vel_xyz.grid(True, alpha=0.3)

    # RFD
    rfd_clean = ev['RFD'].dropna()
    ax_rfd.plot(t_rel.iloc[rfd_clean.index], rfd_clean, color='crimson', linewidth=1.5)
    ax_rfd.axvline(t_rel.iloc[local_peak_f_idx], color='red', ls='--', lw=1.5)
    ax_rfd.set_title('RFD (Rate of Force Development)'); ax_rfd.set_ylabel('N/s'); ax_rfd.set_xlabel('ms')
    ax_rfd.grid(True, alpha=0.3)

    # Tabla
    ax_table.axis('off')
    td = [
        ['Fuerza Pico',       f"{result['Fuerza Pico (N)']:.1f}",          'N'],
        ['Fuerza Neta Pico',  f"{result['Fuerza Neta Pico (N)']:.1f}",      'N'],
        ['Tiempo hasta Pico', f"{result['Tiempo hasta Pico (ms)']:.1f}",    'ms'],
        ['Dur. Contacto',     f"{result['Duracion Contacto (ms)']:.1f}",    'ms'],
        ['Impulso Total',     f"{result['Impulso Total (Ns)']:.3f}",        'N*s'],
        ['Impulso Pico',      f"{result['Impulso hasta Pico (Ns)']:.3f}",   'N*s'],
        ['RFD Max',           f"{result['RFD Max (N/s)']:.0f}",             'N/s'],
        ['F @ 10ms',          f"{result['Fuerza 10ms (N)']:.1f}",           'N'],
        ['RFD 0-10ms',        f"{result['RFD 0-10ms (N/s)']:.0f}",          'N/s'],
        ['F @ 20ms',          f"{result['Fuerza 20ms (N)']:.1f}",           'N'],
        ['Vel. en Impacto',   f"{result['Velocidad en Pico (m/s)']:.3f}",   'm/s'],
        ['Vel. Maxima',       f"{result['Velocidad Max (m/s)']:.3f}",       'm/s'],
        ['Ac. en Pico',       f"{result['Aceleracion en Pico (m/s2)']:.1f}",'m/s2'],
        ['Dur. Golpe (kin)',   f"{result['Duracion Golpe Kin (ms)']:.1f}",   'ms'],
    ]
    tbl = ax_table.table(cellText=td, colLabels=['Metrica', 'Valor', 'Un.'],
                         loc='center', cellLoc='left', colColours=['#e0e0e0']*3)
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1, 1.3)
    ax_table.set_title(f'Metricas - Evento {ev_num}', pad=8)

    plt.savefig(f'event_{ev_num}_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Guardado: event_{ev_num}_analysis.png  |  F={peak_force:.0f}N  V={velocity_at_peak:.2f}m/s')

# Comparacion entre eventos
summary_df = pd.DataFrame(all_results).set_index('Evento')
metrics_to_plot = [
    ('Fuerza Pico (N)',        'Fuerza Pico (N)',           'steelblue'),
    ('Velocidad en Pico (m/s)','Velocidad en Impacto (m/s)','darkorange'),
    ('RFD Max (N/s)',          'RFD Max (N/s)',             'crimson'),
    ('Impulso Total (Ns)',     'Impulso Total (N*s)',        'green'),
    ('Tiempo hasta Pico (ms)', 'Tiempo hasta Pico (ms)',    'purple'),
    ('Duracion Contacto (ms)', 'Duracion Contacto (ms)',    'brown'),
]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for ax, (col, label, color) in zip(axes, metrics_to_plot):
    vals = summary_df[col]
    bars = ax.bar([f'E{i}' for i in summary_df.index], vals, color=color, alpha=0.8, edgecolor='white')
    ax.set_title(label, fontsize=10, fontweight='bold')
    ax.set_ylabel(label, fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.015,
                f'{val:.1f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')
plt.suptitle('Comparacion entre Eventos - K001_J left hand pad', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('events_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print('Guardado: events_comparison.png')

print()
print(summary_df[['Fuerza Pico (N)', 'Velocidad en Pico (m/s)', 'RFD Max (N/s)',
                   'Tiempo hasta Pico (ms)', 'Duracion Contacto (ms)', 'Impulso Total (Ns)']].to_string())
