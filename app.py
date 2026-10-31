"""Gemelo digital consultivo para mantenimiento (TPM) · CNC plasma y torno.

Propuesta académica evaluada por simulación (Monte Carlo). Las señales son
sintéticas y se generan a partir de la línea base del diagnóstico; al implementar
la propuesta, el generador de señales se reemplaza por la lectura del gateway
(MQTT / OPC UA) y el resto de la plataforma no cambia.

Cambios de la versión 1.1 respecto al código original:
  - Sin el nombre de la empresa (títulos, pie, archivos descargables).
  - Nueva pestaña "Réplica virtual" (modelo 3D esquemático con sensores y ficha de datos).
  - Valores por defecto iguales a los del TSP (falla acelerada, 1,000 corridas Monte Carlo).
  - Indicadores con IC 95 %, tabla de criterios de aceptación y pestaña de sensibilidad.
  - Insignias por estado (antes todas salían en verde) y manejo de lecturas perdidas.
  - Resultados en caché (la app ya no recalcula todo al mover un control).
  - Botón para restaurar los valores del TSP (reemplaza al botón "Ejecutar", que no hacía nada).
  - Opción de detección "estimada desde lecturas de sensores" con persistencia real de 4 intervalos.
  - La numérica de la simulación en el modo "versión del TSP" es idéntica a la original.
"""
import json
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

APP_VERSION = "1.1"
APP_TITLE = "Gemelo Digital · TPM"

# Metas del TSP
META_DISPONIBILIDAD = 90.36  # %  (≥)
META_MTBF = 43.86            # h  (>)
META_CALIDAD_DATOS = 95.0    # %  (≥)

SCENARIOS = ["Operación normal", "Desgaste progresivo", "Falla acelerada"]
DETECCION_TSP = "Estado simulado (versión del TSP)"
DETECCION_LECTURAS = "Estimado desde lecturas de sensores"

# Valores por defecto = configuración principal documentada en el TSP
DEFAULTS = {
    "asset_name": "CNC plasma",
    "scenario": "Falla acelerada",
    "policy": "Reactivo",
    "duration": 48,
    "seed": 42,
    "mtbf_gain": 35,
    "mttr_reduction": 35,
    "simulations": 1000,
    "noise_factor": 1.0,
    "missing_rate_pct": 2,
    "intervention_pct": 66,
    "detection": DETECCION_TSP,
}

LEVEL_COLOR = {"Normal": "#34d399", "Advertencia": "#fbbf24", "Crítico": "#fb7185", "Sin dato": "#64748b"}

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .stApp {background: #07111e; color: #e7eef7;}
      [data-testid="stSidebar"] {background: #0a1928; border-right: 1px solid rgba(255,255,255,.08);}
      [data-testid="stMetric"] {background: #0c1d2d; border: 1px solid rgba(255,255,255,.09); border-radius: 14px; padding: 16px;}
      [data-testid="stMetricLabel"] {color: #91a4b8;}
      [data-testid="stMetricValue"] {color: #f6fbff;}
      .block-container {padding-top: 1.6rem; max-width: 1500px;}
      .hero {padding: 20px 24px; border: 1px solid rgba(34,211,238,.20); border-radius: 18px;
             background: radial-gradient(circle at 82% 0%, rgba(34,211,238,.13), transparent 38%), #0b1a2a; margin-bottom: 18px;}
      .eyebrow {color:#67e8f9; font-size:.76rem; font-weight:800; letter-spacing:.18em; text-transform:uppercase;}
      .hero h1 {font-size:2.15rem; margin:.25rem 0; color:#f8fafc;}
      .hero p {color:#94a3b8; margin:0;}
      .demo-note {border:1px solid rgba(251,191,36,.22); background:rgba(251,191,36,.07); color:#fde68a;
                  border-radius:12px; padding:10px 14px; font-size:.88rem; margin-bottom:16px;}
      .status-card {border-radius:16px; padding:18px; border:1px solid rgba(255,255,255,.09); background:#0b1a2a; min-height:155px;}
      .status-normal {box-shadow: inset 4px 0 #34d399;}
      .status-warning {box-shadow: inset 4px 0 #fbbf24;}
      .status-critical {box-shadow: inset 4px 0 #fb7185;}
      .machine-name {color:#f8fafc; font-size:1.18rem; font-weight:750;}
      .machine-meta {color:#7f94a9; font-size:.82rem; margin-top:2px;}
      .status-pill {display:inline-block; margin-top:18px; padding:5px 10px; border-radius:999px; font-size:.75rem; font-weight:800; letter-spacing:.07em;}
      .pill-normal {background:rgba(52,211,153,.13); color:#6ee7b7;}
      .pill-warning {background:rgba(251,191,36,.13); color:#fde68a;}
      .pill-critical {background:rgba(251,113,133,.13); color:#fda4af;}
      .reading {border-radius:14px; padding:14px 16px; background:#0c1d2d; border:1px solid rgba(255,255,255,.09); min-height:112px;}
      .reading-label {color:#91a4b8; font-size:.84rem;}
      .reading-value {color:#f6fbff; font-size:2rem; font-weight:700; line-height:1.25;}
      .reading-meta {color:#8296aa; font-size:.78rem;}
      .legend-dot {display:inline-block; width:10px; height:10px; border-radius:50%; margin:0 5px 0 12px;}
      .section-title {font-size:1.05rem; font-weight:750; color:#f8fafc; margin:8px 0 4px;}
      .small-muted {color:#8296aa; font-size:.83rem;}
      div[data-testid="stDataFrame"] {border:1px solid rgba(255,255,255,.08); border-radius:12px; overflow:hidden;}
      .stButton > button {border-radius:10px; font-weight:700;}
      hr {border-color:rgba(255,255,255,.08)!important;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Activos, variables y umbrales (los umbrales de vibración, temperatura y corriente
# son demostrativos; deben calibrarse con la línea base real y las especificaciones
# del fabricante al implementar la propuesta).
# ---------------------------------------------------------------------------
ASSETS = {
    "CNC plasma": {
        "code": "CNC-01",
        "area": "Corte",
        "scheduled_hours": 300.0,
        "downtime_hours": 40.5,
        "failures": 8,
        "variables": {
            "Vibración (mm/s RMS)": {"base": 1.35, "noise": 0.10, "drift": 4.2, "warning": 3.5, "critical": 5.4, "direction": "high"},
            "Temperatura (°C)": {"base": 43.0, "noise": 0.65, "drift": 39.0, "warning": 68.0, "critical": 78.0, "direction": "high"},
            "Corriente (A)": {"base": 8.8, "noise": 0.22, "drift": 10.5, "warning": 15.0, "critical": 18.0, "direction": "high"},
            "Error de posición (mm)": {"base": 0.028, "noise": 0.003, "drift": 0.115, "warning": 0.10, "critical": 0.12, "direction": "high"},
        },
    },
    "Torno": {
        "code": "TOR-01",
        "area": "Maestranza",
        "scheduled_hours": 220.0,
        "downtime_hours": 29.0,
        "failures": 6,
        "variables": {
            "Vibración (mm/s RMS)": {"base": 1.55, "noise": 0.12, "drift": 4.6, "warning": 4.0, "critical": 5.7, "direction": "high"},
            "Temperatura (°C)": {"base": 45.0, "noise": 0.70, "drift": 37.0, "warning": 70.0, "critical": 79.0, "direction": "high"},
            "Corriente (A)": {"base": 12.2, "noise": 0.30, "drift": 13.0, "warning": 20.0, "critical": 24.0, "direction": "high"},
            "RPM": {"base": 900.0, "noise": 5.5, "drift": -165.0, "warning": 800.0, "critical": 755.0, "direction": "low"},
        },
    },
}

# Qué toma el gemelo de cada variable: sensor, ubicación y origen del dato
# (según la tabla de instrumentación del TSP).
SENSOR_INFO = {
    "CNC plasma": {
        "Vibración (mm/s RMS)": ("Acelerómetro triaxial IEPE o RS-485, IP67", "Servomotor / reductor del eje X", "Gateway (MQTT u OPC UA)"),
        "Temperatura (°C)": ("PT100 clase A con transmisor aislado", "Servomotor / reductor", "Gateway (MQTT u OPC UA)"),
        "Corriente (A)": ("Transductor de núcleo partido, salida aislada", "Accionamiento principal", "Gateway (MQTT u OPC UA)"),
        "Error de posición (mm)": ("Inspección del mantenimiento autónomo (TPM)", "Antorcha y ejes", "Registro de inspección TPM; interfaz del controlador si la exporta"),
    },
    "Torno": {
        "Vibración (mm/s RMS)": ("Acelerómetro triaxial IEPE o RS-485, IP67", "Cabezal", "Gateway (MQTT u OPC UA)"),
        "Temperatura (°C)": ("PT100 clase A con transmisor aislado", "Cabezal / motor", "Gateway (MQTT u OPC UA)"),
        "Corriente (A)": ("Transductor de núcleo partido, salida aislada", "Accionamiento principal", "Gateway (MQTT u OPC UA)"),
        "RPM": ("Sensor inductivo u óptico, salida aislada", "Husillo", "Gateway (MQTT u OPC UA)"),
    },
}


# ---------------------------------------------------------------------------
# Cálculo base
# ---------------------------------------------------------------------------
def base_metrics(asset):
    operating = asset["scheduled_hours"] - asset["downtime_hours"]
    mtbf = operating / asset["failures"]
    mttr = asset["downtime_hours"] / asset["failures"]
    availability = mtbf / (mtbf + mttr) * 100
    return operating, mtbf, mttr, availability


def availability_pct(mtbf, mttr):
    return mtbf / (mtbf + mttr) * 100


def degradation_curve(points, scenario, rng):
    progress = np.linspace(0, 1, points)
    if scenario == "Operación normal":
        curve = 0.02 + 0.05 * progress
    elif scenario == "Desgaste progresivo":
        curve = 0.03 + 0.82 * np.power(progress, 1.75)
    else:
        curve = 0.04 + 1.05 / (1 + np.exp(-10 * (progress - 0.56)))
    return np.clip(curve + rng.normal(0, 0.008, points), 0, 1.10)


def condition_index(observed, asset):
    """Índice de condición estimado SOLO con lecturas: mayor desviación normalizada
    (0 = valor base, 1 = límite crítico) entre las variables, suavizada con mediana de 3."""
    parts = []
    for variable, cfg in asset["variables"].items():
        values = np.asarray(observed[variable], dtype=float)
        if cfg["direction"] == "high":
            parts.append((values - cfg["base"]) / (cfg["critical"] - cfg["base"]))
        else:
            parts.append((cfg["base"] - values) / (cfg["base"] - cfg["critical"]))
    raw = pd.DataFrame(np.vstack(parts).T).max(axis=1, skipna=True)
    return raw.rolling(3, min_periods=1).median().to_numpy()


def persistent_hit(above, intervals=4):
    """Primer instante en que la condición se mantiene durante `intervals` lecturas seguidas."""
    run = pd.Series(above.astype(int)).rolling(intervals).sum().to_numpy()
    hits = np.where(run >= intervals)[0]
    return int(hits[0]) if len(hits) else None


@st.cache_data(show_spinner=False)
def simulate_telemetry(
    asset_name,
    scenario,
    policy,
    duration,
    seed,
    noise_factor=1.0,
    missing_rate=0.0,
    intervention_threshold=0.66,
    detection=DETECCION_TSP,
):
    asset = ASSETS[asset_name]
    rng = np.random.default_rng(seed)
    points = int(duration * 12) + 1  # una lectura cada cinco minutos simulados
    hours = np.linspace(0, duration, points)
    degradation = degradation_curve(points, scenario, rng)

    # El ruido y las pérdidas de datos se sortean en el mismo orden que la versión original.
    noise, missing = {}, {}
    for variable, cfg in asset["variables"].items():
        noise[variable] = rng.normal(0, cfg["noise"] * noise_factor, points)
        missing[variable] = rng.random(points) < missing_rate

    def observe(deg):
        out = {}
        for variable, cfg in asset["variables"].items():
            values = cfg["base"] + deg * cfg["drift"] + noise[variable]
            values = values.copy()
            values[missing[variable]] = np.nan
            out[variable] = values
        return out

    intervention_index = None
    if policy == "Predictivo" and scenario != "Operación normal":
        if detection == DETECCION_TSP:
            # Versión del TSP: cruce del umbral por el estado simulado + 4 intervalos de espera.
            candidates = np.where(degradation >= intervention_threshold)[0]
            if len(candidates):
                intervention_index = int(min(candidates[0] + 4, points - 1))
        else:
            # Detección con lecturas: índice estimado y persistencia real de 4 lecturas seguidas.
            estimate = condition_index(observe(degradation), asset)
            above = np.nan_to_num(estimate, nan=0.0) >= intervention_threshold
            intervention_index = persistent_hit(above, 4)
        if intervention_index is not None:
            residual = degradation[intervention_index]
            degradation[intervention_index:] = np.maximum(
                0.06,
                residual * 0.08 + np.linspace(0, 0.10, points - intervention_index),
            )

    observed_all = observe(degradation)
    df = pd.DataFrame({"Hora simulada": hours, "Degradación": degradation * 100})
    statuses = np.full(points, "Normal", dtype=object)
    alerts = []

    for variable, cfg in asset["variables"].items():
        observed = observed_all[variable]
        df[variable] = observed
        if cfg["direction"] == "high":
            warning_mask = observed >= cfg["warning"]
            critical_mask = observed >= cfg["critical"]
        else:
            warning_mask = observed <= cfg["warning"]
            critical_mask = observed <= cfg["critical"]
        statuses[warning_mask] = "Advertencia"
        statuses[critical_mask] = "Crítico"
        warning_positions = np.where(warning_mask)[0]
        if len(warning_positions):
            idx = int(warning_positions[0])
            alerts.append({
                "Hora": round(float(hours[idx]), 2),
                "Variable": variable,
                "Lectura": round(float(observed[idx]), 3),
                "Umbral": cfg["warning"],
                "Nivel": "Crítico" if critical_mask[idx] else "Advertencia",
            })

    if intervention_index is not None:
        statuses[intervention_index:] = "Normal"
        alerts.append({
            "Hora": round(float(hours[intervention_index]), 2),
            "Variable": "Mantenimiento",
            "Lectura": 0.0,
            "Umbral": 0.0,
            "Nivel": "Intervención ejecutada",
        })

    df["Estado"] = statuses
    df["Índice de condición estimado (%)"] = condition_index(observed_all, asset) * 100
    alerts_df = pd.DataFrame(alerts).sort_values("Hora") if alerts else pd.DataFrame()
    return df, alerts_df, intervention_index


def data_quality(df, asset_name):
    variables = list(ASSETS[asset_name]["variables"].keys())
    expected = len(df) * len(variables)
    observed = int(df[variables].notna().sum().sum())
    return observed / expected * 100 if expected else 0.0


@st.cache_data(show_spinner=False)
def experiment_matrix(asset_name, duration, seed, noise_factor, missing_rate, intervention_threshold, detection):
    rows = []
    for scenario_name in SCENARIOS:
        for policy_name in ["Reactivo", "Predictivo"]:
            trial, trial_alerts, trial_intervention = simulate_telemetry(
                asset_name,
                scenario_name,
                policy_name,
                duration,
                seed + len(rows) * 17,
                noise_factor,
                missing_rate,
                intervention_threshold,
                detection,
            )
            critical = trial.index[trial["Estado"] == "Crítico"].tolist()
            predictive_alerts = (
                trial_alerts[trial_alerts["Nivel"] != "Intervención ejecutada"]
                if not trial_alerts.empty and "Nivel" in trial_alerts.columns
                else pd.DataFrame()
            )
            first_warning = float(predictive_alerts.iloc[0]["Hora"]) if not predictive_alerts.empty else np.nan
            critical_hour = float(trial.loc[critical[0], "Hora simulada"]) if critical else np.nan
            rows.append({
                "Escenario": scenario_name,
                "Política": policy_name,
                "Calidad de datos (%)": round(data_quality(trial, asset_name), 2),
                "Primera alerta (h)": round(first_warning, 2) if not np.isnan(first_warning) else np.nan,
                "Condición crítica (h)": round(critical_hour, 2) if not np.isnan(critical_hour) else np.nan,
                "Intervención (h)": round(float(trial.loc[trial_intervention, "Hora simulada"]), 2)
                if trial_intervention is not None else np.nan,
                "Salud final (%)": round(max(0.0, 100 - float(trial.iloc[-1]["Degradación"])), 2),
            })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def monte_carlo(asset_name, mtbf_gain, mttr_reduction, simulations, seed):
    asset = ASSETS[asset_name]
    _, base_mtbf, base_mttr, _ = base_metrics(asset)
    rng = np.random.default_rng(seed + 99)

    # La dispersión (forma gamma 9 y 8) es un supuesto del modelo, no se estimó de datos.
    reactive_mtbf = rng.gamma(shape=9.0, scale=base_mtbf / 9.0, size=simulations)
    reactive_mttr = rng.gamma(shape=8.0, scale=base_mttr / 8.0, size=simulations)
    predictive_mtbf = rng.gamma(shape=9.0, scale=(base_mtbf * (1 + mtbf_gain / 100)) / 9.0, size=simulations)
    predictive_mttr = rng.gamma(shape=8.0, scale=(base_mttr * (1 - mttr_reduction / 100)) / 8.0, size=simulations)

    reactive_av = reactive_mtbf / (reactive_mtbf + reactive_mttr) * 100
    predictive_av = predictive_mtbf / (predictive_mtbf + predictive_mttr) * 100
    reactive_down = asset["scheduled_hours"] * (1 - reactive_av / 100)
    predictive_down = asset["scheduled_hours"] * (1 - predictive_av / 100)

    return pd.DataFrame({
        "Política": ["Reactiva"] * simulations + ["Predictiva"] * simulations,
        "Disponibilidad (%)": np.concatenate([reactive_av, predictive_av]),
        "Horas de parada": np.concatenate([reactive_down, predictive_down]),
        "MTBF (h)": np.concatenate([reactive_mtbf, predictive_mtbf]),
        "MTTR (h)": np.concatenate([reactive_mttr, predictive_mttr]),
    })


@st.cache_data(show_spinner=False)
def seed_robustness(asset_name, mtbf_gain, mttr_reduction, simulations, n_seeds=100):
    """Repite el Monte Carlo con otras semillas para ver cuánto cambia el promedio."""
    rows = []
    for s in range(1, n_seeds + 1):
        sample = monte_carlo(asset_name, mtbf_gain, mttr_reduction, simulations, s)
        pred = sample[sample["Política"] == "Predictiva"]
        rows.append({
            "Semilla": s,
            "MTBF (h)": pred["MTBF (h)"].mean(),
            "MTTR (h)": pred["MTTR (h)"].mean(),
            "Disponibilidad (%)": pred["Disponibilidad (%)"].mean(),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Estado de las lecturas
# ---------------------------------------------------------------------------
def reading_level(value, cfg):
    if value is None or np.isnan(value):
        return "Sin dato"
    if cfg["direction"] == "high":
        return "Crítico" if value >= cfg["critical"] else "Advertencia" if value >= cfg["warning"] else "Normal"
    return "Crítico" if value <= cfg["critical"] else "Advertencia" if value <= cfg["warning"] else "Normal"


def last_valid(series, moment):
    window = series.iloc[: moment + 1].dropna()
    if window.empty:
        return np.nan, None
    return float(window.iloc[-1]), moment - int(window.index[-1])


def value_text(variable, value):
    if value is None or np.isnan(value):
        return "—"
    decimals = 3 if "posición" in variable else 1
    return f"{value:.{decimals}f}"


def worst_level(levels):
    order = ["Crítico", "Advertencia", "Normal", "Sin dato"]
    for item in order:
        if item in levels:
            return item
    return "Sin dato"


# ---------------------------------------------------------------------------
# Componentes visuales
# ---------------------------------------------------------------------------
def telemetry_figure(df, asset_name, hour=None):
    variables = list(ASSETS[asset_name]["variables"].keys())
    colors = ["#22d3ee", "#fbbf24", "#a78bfa", "#34d399"]
    fig = make_subplots(rows=2, cols=2, subplot_titles=variables, vertical_spacing=0.17)
    for index, variable in enumerate(variables):
        row, col = index // 2 + 1, index % 2 + 1
        cfg = ASSETS[asset_name]["variables"][variable]
        fig.add_trace(
            go.Scatter(x=df["Hora simulada"], y=df[variable], mode="lines", name=variable,
                       line=dict(color=colors[index], width=2.4), showlegend=False),
            row=row, col=col,
        )
        fig.add_hline(y=cfg["warning"], line_dash="dash", line_color="#fbbf24", opacity=.75, row=row, col=col)
        fig.add_hline(y=cfg["critical"], line_dash="dot", line_color="#fb7185", opacity=.80, row=row, col=col)
        if hour is not None:
            fig.add_vline(x=hour, line_color="#cbd5e1", line_width=1, opacity=.55, row=row, col=col)
    fig.update_layout(height=560, margin=dict(l=20, r=20, t=55, b=20), paper_bgcolor="#0b1a2a",
                      plot_bgcolor="#081522", font_color="#b8c7d9", hovermode="x unified")
    fig.update_xaxes(title_text="Horas", gridcolor="rgba(255,255,255,.06)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,.06)")
    return fig


def status_card(asset_name, status, health, hour):
    css_status = {"Normal": "normal", "Advertencia": "warning", "Crítico": "critical"}[status]
    label = {"Normal": "OPERACIÓN NORMAL", "Advertencia": "REQUIERE ATENCIÓN", "Crítico": "RIESGO DE FALLA"}[status]
    asset = ASSETS[asset_name]
    st.markdown(
        f'<div class="status-card status-{css_status}">'
        f'<div class="machine-name">{asset_name} · {asset["code"]}</div>'
        f'<div class="machine-meta">Área de {asset["area"]} · Gemelo simulado · hora {hour:.2f}</div>'
        f'<span class="status-pill pill-{css_status}">{label}</span>'
        f'<div class="small-muted" style="margin-top:12px">Índice de salud simulado: '
        f'<b style="color:#f8fafc">{health:.1f}%</b></div></div>',
        unsafe_allow_html=True,
    )


def reading_card(variable, value, cfg, age_steps):
    level = reading_level(value, cfg)
    color = LEVEL_COLOR[level]
    note = ""
    if age_steps:
        note = f'<div class="reading-meta">Dato perdido: última lectura válida hace {age_steps * 5} min</div>'
    st.markdown(
        f'<div class="reading" style="border-left:4px solid {color}">'
        f'<div class="reading-label">{variable}</div>'
        f'<div class="reading-value">{value_text(variable, value)}</div>'
        f'<div class="reading-meta"><b style="color:{color}">{level}</b> · advertencia {cfg["warning"]} · crítico {cfg["critical"]}</div>'
        f'{note}</div>',
        unsafe_allow_html=True,
    )


_LIGHT = dict(ambient=0.62, diffuse=0.85, specular=0.22, roughness=0.55)


def box_mesh(x0, y0, z0, dx, dy, dz, color, name):
    return go.Mesh3d(
        x=x0 + dx * np.array([0, 0, 1, 1, 0, 0, 1, 1]),
        y=y0 + dy * np.array([0, 1, 1, 0, 0, 1, 1, 0]),
        z=z0 + dz * np.array([0, 0, 0, 0, 1, 1, 1, 1]),
        i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
        j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
        k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
        color=color, name=name, hoverinfo="name", flatshading=True, lighting=_LIGHT, showlegend=False,
    )


def cylinder_mesh(x0, length, radius, yc, zc, color, name, n=40):
    """Cilindro con eje paralelo a X, con tapas."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ys, zs = yc + radius * np.cos(t), zc + radius * np.sin(t)
    x = np.r_[np.full(n, x0), np.full(n, x0 + length), [x0, x0 + length]]
    y = np.r_[ys, ys, [yc, yc]]
    z = np.r_[zs, zs, [zc, zc]]
    c0, c1 = 2 * n, 2 * n + 1
    i, j, k = [], [], []
    for a in range(n):
        b = (a + 1) % n
        i += [a, b, c0, c1]
        j += [b, b + n, b, a + n]
        k += [a + n, a + n, a, b + n]
    return go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color=color, name=name, hoverinfo="name",
                     flatshading=True, lighting=_LIGHT, showlegend=False)


def replica_figure(asset_name, readings):
    """Modelo 3D esquemático (no a escala real). El color de las partes móviles y de los
    marcadores de sensores refleja el estado de las lecturas en el instante elegido."""
    asset = ASSETS[asset_name]
    levels = {v: reading_level(readings[v][0], asset["variables"][v]) for v in asset["variables"]}
    equipment_color = LEVEL_COLOR[worst_level(list(levels.values()))]
    steel, dark, cabinet = "#94a3b8", "#475569", "#1e3a5f"

    if asset_name == "CNC plasma":
        parts = [
            box_mesh(0, 0, 500, 3000, 1500, 80, "#64748b", "Mesa de corte"),
            *[box_mesh(x, y, 0, 120, 120, 500, dark, "Pata") for x in (60, 2820) for y in (60, 1320)],
            box_mesh(-100, -160, 580, 3200, 100, 80, steel, "Riel eje X (izquierdo)"),
            box_mesh(-100, 1560, 580, 3200, 100, 80, steel, "Riel eje X (derecho)"),
            box_mesh(1150, -180, 660, 300, 140, 160, equipment_color, "Carro de pórtico (izquierdo)"),
            box_mesh(1150, 1540, 660, 300, 140, 160, equipment_color, "Carro de pórtico (derecho)"),
            box_mesh(1200, -180, 820, 200, 1880, 140, equipment_color, "Puente (pórtico)"),
            box_mesh(1150, 600, 740, 300, 260, 220, equipment_color, "Carro de antorcha"),
            box_mesh(1255, 700, 600, 50, 60, 140, "#f97316", "Antorcha plasma"),
            box_mesh(3300, 100, 0, 500, 500, 1100, cabinet, "Gabinete de adquisición (DAQ y gateway)"),
        ]
        sensor_points = [
            ("Vibración (mm/s RMS)", "Vibración", (1300, -200, 740)),
            ("Temperatura (°C)", "Temperatura", (1350, 1740, 740)),
            ("Corriente (A)", "Corriente", (3550, 350, 1150)),
            ("Error de posición (mm)", "Error de posición (inspección TPM)", (1280, 730, 590)),
        ]
        eye = dict(x=1.35, y=-1.65, z=0.95)
    else:
        parts = [
            box_mesh(0, 0, 0, 2400, 600, 160, "#64748b", "Bancada"),
            box_mesh(150, 50, -350, 400, 500, 350, dark, "Pedestal"),
            box_mesh(1800, 50, -350, 400, 500, 350, dark, "Pedestal"),
            box_mesh(0, 0, 160, 600, 600, 520, equipment_color, "Cabezal (motor y husillo)"),
            cylinder_mesh(600, 140, 230, 300, 490, steel, "Plato de sujeción"),
            cylinder_mesh(740, 1000, 70, 300, 490, "#cbd5e1", "Pieza"),
            cylinder_mesh(1740, 160, 30, 300, 490, steel, "Punto del contrapunto"),
            box_mesh(1900, 150, 160, 350, 300, 400, steel, "Contrapunto"),
            box_mesh(1000, -50, 160, 320, 700, 120, equipment_color, "Carro longitudinal"),
            box_mesh(1050, 330, 280, 220, 300, 100, steel, "Carro transversal"),
            box_mesh(1100, 380, 380, 140, 200, 90, dark, "Portaherramienta"),
            box_mesh(1150, 380, 380, 40, 40, 110, "#f97316", "Herramienta"),
            box_mesh(2600, 100, 0, 500, 500, 1100, cabinet, "Gabinete de adquisición (DAQ y gateway)"),
        ]
        sensor_points = [
            ("Vibración (mm/s RMS)", "Vibración", (300, -40, 620)),
            ("Temperatura (°C)", "Temperatura", (500, 640, 450)),
            ("Corriente (A)", "Corriente", (2850, 350, 1150)),
            ("RPM", "RPM", (670, 300, 770)),
        ]
        eye = dict(x=1.2, y=-1.75, z=0.85)

    sx, sy, sz, text, hover, colors = [], [], [], [], [], []
    for variable, label, (x, y, z) in sensor_points:
        value, _ = readings[variable]
        level = levels[variable]
        sx.append(x); sy.append(y); sz.append(z)
        text.append(label)
        hover.append(f"{variable}: {value_text(variable, value)} · {level}")
        colors.append(LEVEL_COLOR[level])
    sensors = go.Scatter3d(
        x=sx, y=sy, z=sz, mode="markers+text", text=text, textposition="top center",
        textfont=dict(color="#e2e8f0", size=11), hovertext=hover, hoverinfo="text", showlegend=False,
        marker=dict(size=8, color=colors, symbol="diamond", line=dict(width=1, color="#0f172a")),
    )
    fig = go.Figure(parts + [sensors])
    fig.update_layout(
        height=580, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="#0b1a2a", font_color="#b8c7d9",
        scene=dict(aspectmode="data", bgcolor="#081522", camera=dict(eye=eye),
                   xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
    )
    return fig, levels


def sensitivity_figure(asset_name, mtbf_gain, mttr_reduction):
    _, M, R, _ = base_metrics(ASSETS[asset_name])
    gains = np.arange(0, 61, 5)
    reductions = np.arange(0, 61, 5)
    z = np.array([[availability_pct(M * (1 + g / 100), R * (1 - r / 100)) for g in gains] for r in reductions])
    fig = go.Figure(go.Heatmap(z=z, x=gains, y=reductions, colorscale="Viridis",
                               colorbar=dict(title="Disp. (%)"),
                               hovertemplate="MTBF +%{x} %<br>MTTR −%{y} %<br>Disponibilidad %{z:.2f} %<extra></extra>"))
    fig.add_trace(go.Contour(z=z, x=gains, y=reductions, showscale=False, hoverinfo="skip",
                             contours=dict(coloring="none", start=META_DISPONIBILIDAD, end=META_DISPONIBILIDAD,
                                           size=1, showlabels=True),
                             line=dict(color="#fb7185", width=3)))
    fig.add_trace(go.Scatter(x=[mtbf_gain], y=[mttr_reduction], mode="markers", name="Supuesto actual",
                             marker=dict(size=14, color="white", line=dict(color="black", width=2))))
    fig.update_layout(height=440, paper_bgcolor="#0b1a2a", plot_bgcolor="#081522", font_color="#b8c7d9",
                      margin=dict(l=20, r=20, t=30, b=20),
                      xaxis_title="Incremento del MTBF (%)", yaxis_title="Reducción del MTTR (%)",
                      legend=dict(orientation="h", y=1.1))
    return fig


# ---------------------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------------------
for _key, _value in DEFAULTS.items():
    st.session_state.setdefault(_key, _value)


def reset_defaults():
    for key, value in DEFAULTS.items():
        st.session_state[key] = value


with st.sidebar:
    st.markdown("## ⚙️ Configuración")
    asset_name = st.selectbox("Activo digital", list(ASSETS.keys()), key="asset_name")
    scenario = st.selectbox("Escenario", SCENARIOS, key="scenario")
    policy = st.radio("Política de mantenimiento", ["Reactivo", "Predictivo"], horizontal=True, key="policy")
    duration = st.slider("Duración simulada (horas)", 8, 72, step=4, key="duration")
    seed = st.number_input("Semilla del experimento", min_value=1, max_value=9999, key="seed")
    st.divider()
    st.markdown("#### Supuestos TO-BE")
    mtbf_gain = st.slider("Incremento proyectado del MTBF (%)", 5, 60, step=1, key="mtbf_gain")
    mttr_reduction = st.slider("Reducción proyectada del MTTR (%)", 5, 60, step=1, key="mttr_reduction")
    simulations = st.select_slider("Corridas Monte Carlo", options=[100, 250, 500, 1000], key="simulations")
    st.divider()
    st.markdown("#### Pruebas de robustez")
    noise_factor = st.slider("Ruido del sensor", 0.5, 3.0, step=0.1, key="noise_factor")
    missing_rate_pct = st.slider("Pérdida simulada de datos (%)", 0, 20, step=1, key="missing_rate_pct")
    intervention_pct = st.slider("Umbral de intervención (% de degradación)", 45, 85, step=1, key="intervention_pct")
    detection = st.selectbox(
        "Método de detección", [DETECCION_TSP, DETECCION_LECTURAS], key="detection",
        help="'Versión del TSP': el umbral se evalúa sobre el estado interno del simulador. "
             "'Estimado desde lecturas': el gemelo estima la condición solo con las lecturas de los sensores "
             "y exige 4 lecturas seguidas por encima del umbral.",
    )
    st.button("↺ Restaurar valores del TSP", on_click=reset_defaults, width="stretch")
    st.caption("Los valores por defecto reproducen la configuración principal del TSP. "
               "Los supuestos pueden modificarse durante la sustentación para el análisis de sensibilidad.")

intervention_threshold = intervention_pct / 100
missing_rate = missing_rate_pct / 100

# ---------------------------------------------------------------------------
# Cálculos principales (en caché)
# ---------------------------------------------------------------------------
sim_args = (duration, int(seed), float(noise_factor), float(missing_rate), float(intervention_threshold), detection)
df_re, alerts_re, idx_re = simulate_telemetry(asset_name, scenario, "Reactivo", *sim_args)
df_pr, alerts_pr, idx_pr = simulate_telemetry(asset_name, scenario, "Predictivo", *sim_args)
df, alerts, intervention_index = (df_re, alerts_re, idx_re) if policy == "Reactivo" else (df_pr, alerts_pr, idx_pr)

mc = monte_carlo(asset_name, mtbf_gain, mttr_reduction, int(simulations), int(seed))
experiments = experiment_matrix(asset_name, *sim_args)
asset = ASSETS[asset_name]
operating, base_mtbf, base_mttr, base_availability = base_metrics(asset)

det_mtbf = base_mtbf * (1 + mtbf_gain / 100)
det_mttr = base_mttr * (1 - mttr_reduction / 100)
det_availability = availability_pct(det_mtbf, det_mttr)

if policy == "Predictivo":
    projected_mtbf, projected_mttr = det_mtbf, det_mttr
else:
    projected_mtbf, projected_mttr = base_mtbf, base_mttr
projected_availability = availability_pct(projected_mtbf, projected_mttr)
projected_downtime = asset["scheduled_hours"] * (1 - projected_availability / 100)
avoided_downtime = max(0.0, asset["downtime_hours"] - projected_downtime)
completeness = data_quality(df, asset_name)

# ---------------------------------------------------------------------------
# Encabezado
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Industria 4.0 · TPM · Gemelo digital consultivo</div>
      <h1>Gemelo digital de mantenimiento</h1>
      <p>Representación virtual del CNC plasma y del torno, con telemetría simulada, detección de degradación y evaluación de la política de mantenimiento.</p>
    </div>
    <div class="demo-note"><b>SIMULACIÓN ACADÉMICA:</b> las señales y los resultados TO-BE son escenarios calculados; no provienen de sensores instalados.</div>
    """,
    unsafe_allow_html=True,
)

hour_selected = st.slider(
    "Instante de la simulación (hora simulada)", 0.0, float(duration), float(duration), step=1 / 12, format="%.2f h"
)
moment = int(min(max(round(hour_selected * 12), 0), len(df) - 1))
hour_now = float(df.iloc[moment]["Hora simulada"])

col_status, col_health, col_risk, col_data = st.columns([1.6, 1, 1, 1])
status_now = str(df.iloc[moment]["Estado"])
health_now = float(min(100.0, max(0.0, 100 - float(df.iloc[moment]["Degradación"]))))
with col_status:
    status_card(asset_name, status_now, health_now, hour_now)
with col_health:
    st.metric("Disponibilidad proyectada", f"{projected_availability:.2f}%", f"{projected_availability - base_availability:+.2f} pp")
    st.metric("MTBF proyectado", f"{projected_mtbf:.2f} h", f"{projected_mtbf - base_mtbf:+.2f} h")
with col_risk:
    st.metric("MTTR proyectado", f"{projected_mttr:.2f} h", f"{projected_mttr - base_mttr:+.2f} h", delta_color="inverse")
    st.metric("Parada potencial evitada", f"{avoided_downtime:.2f} h", "Periodo diagnóstico", delta_color="off")
with col_data:
    st.metric("Calidad de datos", f"{completeness:.2f}%", f"Meta ≥ {META_CALIDAD_DATOS:.0f}%", delta_color="off")
    st.metric("Registros simulados", f"{len(df):,}", "Cada 5 min", delta_color="off")

(tab_live, tab_replica, tab_compare, tab_sens, tab_alerts, tab_tests, tab_impl, tab_method) = st.tabs([
    "📡 Telemetría", "🧊 Réplica virtual", "📊 Indicadores", "📈 Sensibilidad",
    "⚠️ Alertas", "🧪 Pruebas", "🛠️ Implementación", "📘 Metodología",
])

# lecturas en el instante elegido (con la última lectura válida si el dato se perdió)
readings = {}
for variable in asset["variables"]:
    readings[variable] = last_valid(df[variable], moment)

# ---------------------------------------------------------------------------
# Telemetría
# ---------------------------------------------------------------------------
with tab_live:
    st.markdown('<div class="section-title">Comportamiento virtual del activo</div>', unsafe_allow_html=True)
    st.caption("Línea amarilla: umbral de advertencia · Línea roja: umbral crítico · Línea gris vertical: instante elegido")
    st.plotly_chart(telemetry_figure(df, asset_name, hour_now), width="stretch", config={"displaylogo": False})
    columns = st.columns(4)
    for column, variable in zip(columns, asset["variables"]):
        value, age = readings[variable]
        with column:
            reading_card(variable, value, asset["variables"][variable], age)
    st.download_button(
        "⬇ Descargar telemetría CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"telemetria_{asset['code'].lower()}.csv",
        mime="text/csv",
    )

# ---------------------------------------------------------------------------
# Réplica virtual
# ---------------------------------------------------------------------------
with tab_replica:
    st.markdown('<div class="section-title">Réplica virtual del equipo</div>', unsafe_allow_html=True)
    st.caption("Modelo 3D esquemático, no a escala real. Los marcadores indican los puntos de medición definidos en la "
               "propuesta de instrumentación; su color refleja el estado de la lectura simulada en el instante elegido.")
    left, right = st.columns([2.1, 1])
    with left:
        figure, levels = replica_figure(asset_name, readings)
        st.plotly_chart(figure, width="stretch", config={"displaylogo": False})
        legend = "".join(
            f'<span class="legend-dot" style="background:{LEVEL_COLOR[name]}"></span>{name}'
            for name in ["Normal", "Advertencia", "Crítico", "Sin dato"]
        )
        st.markdown(f'<div class="small-muted">Estado del punto de medición:{legend}</div>', unsafe_allow_html=True)
    with right:
        estimate_now = float(df.iloc[moment]["Índice de condición estimado (%)"])
        st.markdown('<div class="section-title">Ficha digital del equipo</div>', unsafe_allow_html=True)
        ficha = pd.DataFrame({
            "Atributo": ["Identificación", "Tipo de elemento", "Área", "Estado actual", "Hora simulada",
                         "Índice de salud (simulador)", "Condición estimada desde lecturas",
                         "Fuente de datos", "Actualización", "Rol del gemelo"],
            "Valor": [asset["code"], "Equipo (elemento de manufactura observable)", asset["area"], status_now,
                      f"{hour_now:.2f} h", f"{health_now:.1f} %", f"{estimate_now:.1f} %",
                      "Señal sintética (simulación)", "Cada 5 min (casi en tiempo real)",
                      "Consultivo: propone la OT; decide el jefe de Mantenimiento"],
        })
        st.dataframe(ficha, hide_index=True, width="stretch")
    st.markdown('<div class="section-title">Qué datos toma el gemelo</div>', unsafe_allow_html=True)
    rows = []
    for variable, cfg in asset["variables"].items():
        sensor, place, origin = SENSOR_INFO[asset_name][variable]
        rows.append({
            "Variable": variable,
            "Sensor (implementación)": sensor,
            "Ubicación": place,
            "Umbral de advertencia": cfg["warning"],
            "Umbral crítico": cfg["critical"],
            "Origen en esta simulación": "Señal sintética cada 5 min, identificada como simulada",
            "Origen en la implementación": origin,
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Los umbrales son demostrativos y se calibrarán con la línea base real y las especificaciones del fabricante. "
               "Las cantidades de instrumentación están en la propuesta de implementación del TSP.")

# ---------------------------------------------------------------------------
# Indicadores
# ---------------------------------------------------------------------------
with tab_compare:
    st.markdown('<div class="section-title">Resultados de la simulación Monte Carlo</div>', unsafe_allow_html=True)
    st.caption(f"Resultado de {simulations} corridas por política. El incremento del MTBF y la reducción del MTTR "
               "son supuestos configurables, no resultados medidos.")
    metrics_cols = ["Disponibilidad (%)", "Horas de parada", "MTBF (h)", "MTTR (h)"]
    grouped = mc.groupby("Política")[metrics_cols]
    means, stds = grouped.mean(), grouped.std()
    summary = means.copy().astype(object)
    for policy_name in means.index:
        for column_name in metrics_cols:
            half_width = 1.96 * stds.loc[policy_name, column_name] / np.sqrt(simulations)
            summary.loc[policy_name, column_name] = f"{means.loc[policy_name, column_name]:.2f} ± {half_width:.2f}"
    st.dataframe(summary, width="stretch")
    st.caption("Promedio ± IC 95 % del promedio. El IC mide el error de muestreo del Monte Carlo, no la incertidumbre del supuesto de mejora.")

    fig_box = make_subplots(rows=1, cols=2, subplot_titles=["Distribución de disponibilidad", "Distribución de horas de parada"])
    colors = {"Reactiva": "#64748b", "Predictiva": "#22d3ee"}
    for policy_name in ["Reactiva", "Predictiva"]:
        subset = mc[mc["Política"] == policy_name]
        fig_box.add_trace(go.Box(y=subset["Disponibilidad (%)"], name=policy_name, marker_color=colors[policy_name], showlegend=False), row=1, col=1)
        fig_box.add_trace(go.Box(y=subset["Horas de parada"], name=policy_name, marker_color=colors[policy_name], showlegend=False), row=1, col=2)
    fig_box.add_hline(y=META_DISPONIBILIDAD, line_dash="dash", line_color="#34d399", row=1, col=1)
    fig_box.update_layout(height=430, paper_bgcolor="#0b1a2a", plot_bgcolor="#081522", font_color="#b8c7d9", margin=dict(l=20, r=20, t=55, b=20))
    fig_box.update_yaxes(gridcolor="rgba(255,255,255,.06)")
    st.plotly_chart(fig_box, width="stretch", config={"displaylogo": False})
    st.caption(f"Línea verde: meta de disponibilidad ({META_DISPONIBILIDAD} %).")

    st.markdown('<div class="section-title">Criterios de aceptación (en simulación)</div>', unsafe_allow_html=True)
    predictive = mc[mc["Política"] == "Predictiva"]
    mc_availability = predictive["Disponibilidad (%)"].mean()
    mc_mtbf = predictive["MTBF (h)"].mean()
    first_alert_row = alerts_pr[alerts_pr["Nivel"] != "Intervención ejecutada"] if not alerts_pr.empty else alerts_pr
    first_alert_hour = float(first_alert_row.iloc[0]["Hora"]) if len(first_alert_row) else np.nan
    critical_reactive = df_re.index[df_re["Estado"] == "Crítico"].tolist()
    critical_hour_reactive = float(df_re.loc[critical_reactive[0], "Hora simulada"]) if critical_reactive else np.nan
    predictive_critical = bool((df_pr["Estado"] == "Crítico").any())
    intervention_hour = float(df_pr.loc[idx_pr, "Hora simulada"]) if idx_pr is not None else np.nan

    def verdict(ok):
        return "Cumple" if ok else "No cumple"

    criteria = [
        ("Calidad de datos", f"≥ {META_CALIDAD_DATOS:.0f} %", f"{data_quality(df_pr, asset_name):.2f} %", verdict(data_quality(df_pr, asset_name) >= META_CALIDAD_DATOS)),
        ("Disponibilidad determinística (supuestos actuales)", f"≥ {META_DISPONIBILIDAD} %", f"{det_availability:.2f} %", verdict(det_availability >= META_DISPONIBILIDAD)),
        ("Disponibilidad media Monte Carlo (predictiva)", f"≥ {META_DISPONIBILIDAD} %", f"{mc_availability:.2f} %", verdict(mc_availability >= META_DISPONIBILIDAD)),
        ("MTBF determinístico (supuestos actuales)", f"> {META_MTBF} h", f"{det_mtbf:.2f} h", verdict(det_mtbf > META_MTBF)),
        ("MTBF medio Monte Carlo (predictiva)", f"> {META_MTBF} h", f"{mc_mtbf:.2f} h", verdict(mc_mtbf > META_MTBF)),
    ]
    if scenario != "Operación normal":
        criteria += [
            ("Detección antes de criticidad", "Alerta previa", f"Primera alerta: {first_alert_hour:.2f} h" if not np.isnan(first_alert_hour) else "Sin alerta", verdict(not np.isnan(first_alert_hour))),
            ("Intervención virtual", "Evitar criticidad simulada",
             f"Intervención: {intervention_hour:.2f} h" if not np.isnan(intervention_hour) else "Sin intervención",
             verdict(not np.isnan(intervention_hour) and not predictive_critical)),
        ]
    criteria.append(("Reproducibilidad", "Mismos parámetros", f"Semilla {int(seed)}", "Cumple"))
    st.dataframe(pd.DataFrame(criteria, columns=["Criterio", "Meta", "Resultado", "Estado"]), hide_index=True, width="stretch")
    if not np.isnan(critical_hour_reactive) and not np.isnan(first_alert_hour):
        st.caption(f"Con la política reactiva, la condición crítica se alcanza {critical_hour_reactive - first_alert_hour:.2f} h "
                   "después de la primera alerta. Todo cumplimiento corresponde al entorno simulado.")

    left_box, right_box = st.columns(2)
    with left_box:
        st.markdown("**Lo que la simulación sí evalúa**")
        st.markdown(
            "- Que la lógica de alerta detecta la degradación antes del punto crítico.\n"
            "- Cuánto cambian la disponibilidad y las horas de parada si el MTBF y el MTTR mejoran.\n"
            "- La sensibilidad de los indicadores al ruido, a la pérdida de datos y al umbral."
        )
    with right_box:
        st.markdown("**Lo que es un supuesto**")
        st.markdown(
            "- El incremento del MTBF y la reducción del MTTR: se configuran, no se calculan a partir de la detección.\n"
            "- La dispersión del MTBF y del MTTR en el Monte Carlo (formas gamma 9 y 8).\n"
            "- Los umbrales de vibración, temperatura y corriente (demostrativos)."
        )

# ---------------------------------------------------------------------------
# Sensibilidad
# ---------------------------------------------------------------------------
with tab_sens:
    st.markdown('<div class="section-title">Qué supuestos hacen falta para cumplir las metas</div>', unsafe_allow_html=True)
    st.caption("Disponibilidad determinística = MTBF / (MTBF + MTTR). La línea roja marca la meta de disponibilidad; el punto blanco, el supuesto actual.")
    st.plotly_chart(sensitivity_figure(asset_name, mtbf_gain, mttr_reduction), width="stretch", config={"displaylogo": False})

    needed_gain = (META_MTBF / base_mtbf - 1) * 100
    needed_reduction = (1 - (base_mtbf * (1 - META_DISPONIBILIDAD / 100) / (META_DISPONIBILIDAD / 100)) / base_mttr) * 100
    required = pd.DataFrame([
        {"Condición": f"Incremento mínimo del MTBF para superar {META_MTBF} h", "Valor requerido": f"{needed_gain:.1f} %",
         "Supuesto actual": f"{mtbf_gain} %", "Estado": verdict(mtbf_gain >= needed_gain)},
        {"Condición": f"Reducción mínima del MTTR para llegar a {META_DISPONIBILIDAD} % de disponibilidad sin mejorar el MTBF",
         "Valor requerido": f"{needed_reduction:.1f} %", "Supuesto actual": f"{mttr_reduction} %", "Estado": verdict(mttr_reduction >= needed_reduction)},
        {"Condición": f"Disponibilidad determinística con los supuestos actuales (meta ≥ {META_DISPONIBILIDAD} %)",
         "Valor requerido": f"{META_DISPONIBILIDAD} %", "Supuesto actual": f"{det_availability:.2f} %", "Estado": verdict(det_availability >= META_DISPONIBILIDAD)},
    ])
    st.dataframe(required, hide_index=True, width="stretch")

    st.markdown('<div class="section-title">Estabilidad del Monte Carlo frente a la semilla</div>', unsafe_allow_html=True)
    robust = seed_robustness(asset_name, mtbf_gain, mttr_reduction, int(simulations))
    share_mtbf = (robust["MTBF (h)"] > META_MTBF).mean() * 100
    share_av = (robust["Disponibilidad (%)"] >= META_DISPONIBILIDAD).mean() * 100
    current = monte_carlo(asset_name, mtbf_gain, mttr_reduction, int(simulations), int(seed))
    current_mtbf = current[current["Política"] == "Predictiva"]["MTBF (h)"].mean()
    percentile = (robust["MTBF (h)"] < current_mtbf).mean() * 100
    c1, c2, c3 = st.columns(3)
    c1.metric("MTBF medio entre 100 semillas", f"{robust['MTBF (h)'].mean():.2f} h", f"Valor esperado: {det_mtbf:.2f} h", delta_color="off")
    c2.metric(f"Semillas con MTBF > {META_MTBF} h", f"{share_mtbf:.0f} %", f"Semilla {int(seed)}: {current_mtbf:.2f} h (percentil {percentile:.0f})", delta_color="off")
    c3.metric(f"Semillas con disponibilidad ≥ {META_DISPONIBILIDAD} %", f"{share_av:.0f} %", f"Promedio: {robust['Disponibilidad (%)'].mean():.2f} %", delta_color="off")
    fig_seed = go.Figure(go.Histogram(x=robust["MTBF (h)"], nbinsx=25, marker_color="#22d3ee"))
    fig_seed.add_vline(x=META_MTBF, line_dash="dash", line_color="#fb7185", annotation_text="Meta MTBF")
    fig_seed.add_vline(x=current_mtbf, line_color="#fbbf24", annotation_text=f"Semilla {int(seed)}", annotation_position="top left")
    fig_seed.update_layout(height=340, paper_bgcolor="#0b1a2a", plot_bgcolor="#081522", font_color="#b8c7d9",
                           margin=dict(l=20, r=20, t=30, b=20), xaxis_title="MTBF predictivo medio de cada semilla (h)", yaxis_title="Semillas")
    st.plotly_chart(fig_seed, width="stretch", config={"displaylogo": False})
    st.caption("Cada barra agrupa semillas distintas. Si el valor esperado del MTBF queda cerca de la meta, que se cumpla o no "
               "depende de la semilla. Conviene reportar el promedio entre semillas y la probabilidad de cumplir, no un solo valor.")

# ---------------------------------------------------------------------------
# Alertas
# ---------------------------------------------------------------------------
with tab_alerts:
    st.markdown('<div class="section-title">Registro de eventos predictivos</div>', unsafe_allow_html=True)
    if alerts.empty:
        st.success("No se detectaron anomalías durante el escenario seleccionado.")
    else:
        st.dataframe(alerts, width="stretch", hide_index=True)
        first_alert = alerts[alerts["Nivel"] != "Intervención ejecutada"]
        if not first_alert.empty:
            alert_hour = float(first_alert.iloc[0]["Hora"])
            critical_rows = df[df["Estado"] == "Crítico"]
            lead_time = max(0.0, float(critical_rows.iloc[0]["Hora simulada"]) - alert_hour) if not critical_rows.empty else duration - alert_hour
            st.info(f"Tiempo de anticipación simulado: **{lead_time:.2f} horas** desde la primera alerta hasta la condición crítica o el final del escenario.")
        if intervention_index is not None:
            st.success(f"El mantenimiento predictivo fue ejecutado virtualmente en la hora {df.iloc[intervention_index]['Hora simulada']:.2f}, evitando que la degradación continuara hasta la falla.")

# ---------------------------------------------------------------------------
# Pruebas
# ---------------------------------------------------------------------------
with tab_tests:
    st.markdown("### Matriz automática de experimentos")
    st.caption("Compara los tres estados de degradación bajo política reactiva y predictiva con la misma configuración de robustez. "
               "Cada experimento usa una semilla distinta (semilla + 17 por experimento).")
    st.dataframe(experiments, width="stretch", hide_index=True)
    st.download_button(
        "⬇ Descargar matriz de pruebas",
        data=experiments.to_csv(index=False).encode("utf-8"),
        file_name=f"pruebas_gemelo_{asset['code'].lower()}.csv",
        mime="text/csv",
    )
    if completeness < META_CALIDAD_DATOS:
        st.error("La completitud es menor a 95 %. El ensayo debe considerarse no conforme hasta corregir la pérdida de datos.")
    else:
        st.success("La calidad de datos cumple el criterio académico propuesto de ≥95 %.")

# ---------------------------------------------------------------------------
# Implementación
# ---------------------------------------------------------------------------
with tab_impl:
    st.markdown("### Qué necesita la empresa para conectar el gemelo")
    st.markdown(
        """
        La plataforma ya contiene la lógica del gemelo: umbrales, detección, política de mantenimiento y decisión.
        Al implementar la propuesta, el generador de señales se reemplaza por una entrada MQTT, OPC UA o CSV proveniente
        del gateway industrial; la interfaz y las reglas no cambian.

        | Capa | CNC plasma | Torno | Requisito común |
        |---|---|---|---|
        | Sensores | Vibración, PT100, corriente; error de posición por inspección TPM | Vibración, PT100, corriente y RPM | Montaje trazable y calibración |
        | Adquisición | DAQ IEPE + módulo industrial | DAQ IEPE + módulo industrial | Gabinete IP65 y fuente 24 VDC |
        | Comunicación | MQTT/OPC UA sobre Ethernet | MQTT/OPC UA sobre Ethernet | Red segmentada, TLS y reloj sincronizado |
        | Datos | Series temporales + estado de eje | Series temporales + estado del cabezal | Base de datos y respaldo |
        | Aplicación | Este dashboard | Este dashboard | Confirmación humana de alertas |
        """
    )
    st.info("Plazos, costos y condiciones de seguridad (IPERC, bloqueo y etiquetado, captura pasiva previa) están en la "
            "propuesta de implementación del trabajo.")
    st.warning("Esta plataforma evalúa la lógica del gemelo con datos simulados. Al implementarla, los umbrales deben "
               "calibrarse con la línea base real del equipo y las especificaciones del fabricante.")

# ---------------------------------------------------------------------------
# Metodología
# ---------------------------------------------------------------------------
with tab_method:
    st.markdown("### Diseño de validación")
    st.markdown(
        """
        1. **Calibración AS-IS:** se utilizan las horas programadas, las horas de parada y las fallas históricas de la empresa.
        2. **Generación de señales:** se simula el crecimiento progresivo de vibración, temperatura, corriente y error de posición o RPM.
        3. **Detección:** cada señal se compara con umbrales de advertencia y críticos.
        4. **Intervención virtual:** la política predictiva ejecuta mantenimiento al detectar degradación sostenida.
        5. **Monte Carlo:** se comparan las distribuciones de MTBF, MTTR, disponibilidad y parada entre ambas políticas.
        """
    )
    baseline_table = pd.DataFrame([
        {
            "Equipo": name,
            "Horas programadas": cfg["scheduled_hours"],
            "Horas de parada": cfg["downtime_hours"],
            "Fallas": cfg["failures"],
            "MTBF (h)": round(base_metrics(cfg)[1], 2),
            "MTTR (h)": round(base_metrics(cfg)[2], 2),
            "Disponibilidad (%)": round(base_metrics(cfg)[3], 2),
        }
        for name, cfg in ASSETS.items()
    ])
    st.dataframe(baseline_table, width="stretch", hide_index=True)
    st.markdown("### Alcance y limitaciones")
    st.markdown(
        """
        - Las señales son sintéticas y se generan a partir de la línea base del diagnóstico; no provienen de sensores instalados.
        - El incremento del MTBF y la reducción del MTTR son supuestos configurables. La simulación no los calcula a partir de la detección.
        - En el método "versión del TSP", el cruce del umbral se evalúa sobre el estado interno del simulador y se agregan 4 intervalos de espera. El método "estimado desde lecturas" usa solo las lecturas y exige 4 seguidas por encima del umbral.
        - La dispersión del Monte Carlo (formas gamma 9 y 8) es un supuesto del modelo.
        - Los umbrales de vibración, temperatura y corriente son demostrativos y deben calibrarse con mediciones normales del equipo, especificaciones del fabricante y evaluación técnica.
        - El modelo 3D es esquemático y no está a escala.
        """
    )

# ---------------------------------------------------------------------------
# Configuración exportable y pie
# ---------------------------------------------------------------------------
config_export = {
    "version_app": APP_VERSION,
    "generado": datetime.now().isoformat(timespec="seconds"),
    "activo": asset_name,
    "escenario": scenario,
    "politica": policy,
    "duracion_h": duration,
    "semilla": int(seed),
    "incremento_mtbf_pct": mtbf_gain,
    "reduccion_mttr_pct": mttr_reduction,
    "corridas_monte_carlo": int(simulations),
    "ruido_sensor": noise_factor,
    "perdida_datos_pct": missing_rate_pct,
    "umbral_intervencion_pct": intervention_pct,
    "metodo_deteccion": detection,
}
with st.sidebar:
    st.download_button(
        "⬇ Descargar configuración (JSON)",
        data=json.dumps(config_export, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="configuracion_simulacion.json",
        mime="application/json",
        width="stretch",
    )

st.markdown("---")
st.caption(f"Propuesta académica de gemelo digital consultivo · Datos simulados · v{APP_VERSION}")
