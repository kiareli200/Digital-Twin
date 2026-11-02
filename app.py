"""Gemelo digital consultivo para mantenimiento (TPM) · CNC plasma y torno.

Funciona con señales simuladas (para evaluar el gemelo) o con lecturas reales cargadas desde un CSV.
Para conectar un gateway (MQTT / OPC UA), basta con volcar sus lecturas al formato del CSV de entrada
o reemplazar `parse_csv` por un lector del gateway: el análisis posterior no cambia.

Configuración por variables de entorno (ver README): GEMELO_DATA_DIR, GEMELO_CONFIG, GEMELO_USERS,
GEMELO_SMTP_*, GEMELO_MAIL_TO, GEMELO_WEBHOOK_URL.

Ejecución:  streamlit run app.py
"""
from contextlib import contextmanager
import copy
from dataclasses import dataclass, replace
from datetime import datetime
from email.message import EmailMessage
import hashlib
import hmac
import io
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import smtplib
import sqlite3
import time
from types import SimpleNamespace
from typing import Callable, NamedTuple, Optional
import unicodedata
import urllib.request

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


# ===========================================================================
# CONFIG
# ===========================================================================

APP_VERSION = "2.0"
APP_TITLE = "Gemelos Digitales · TPM"

# Metas de aceptación
META_DISPONIBILIDAD = 90.36  # %  (≥)
META_MTBF = 43.86            # h  (>)
META_CALIDAD_DATOS = 95.0    # %  (≥)

# Resolución temporal: una lectura cada cinco minutos simulados
SAMPLES_PER_HOUR = 12
THERMAL_INERTIA_SAMPLES = 24    # ventana de la inercia térmica de la carga (2 h)
PERSISTENCE_INTERVALS = 4       # lecturas seguidas por encima del umbral para proponer intervenir
LEARN_H = 4.0                   # horas con las que el gemelo aprende (o recalibra) el perfil normal

# Semillas base de los análisis con muchas trayectorias
SEED_FAILING = 5000
SEED_NORMAL = 9000
SEED_STEP_EXPERIMENT = 17

POLICIES = ("Reactivo", "Predictivo")
SCENARIOS = ["Operación normal", "Desgaste progresivo", "Falla acelerada"]
DETECCION_ESTADO = "Estado interno del simulador"
DETECCION_LECTURAS = "Estimado desde lecturas de sensores"
CAPTURA_CONTINUA = "Continua cada 5 min"
CAPTURA_INSPECCION = "Por inspección TPM"

# Vistas principales (navegación)
SIMULACION = "Simulación"
DATOS_REALES = "Datos reales (CSV)"
DATA_SOURCES = [SIMULACION, DATOS_REALES]

# Vistas principales (navegación). La de configuración solo la ve quien tiene permiso de configurar.
V_TELEMETRIA, V_REPLICA, V_ALERTAS = "📡 Telemetría", "🧊 Réplica virtual", "⚠️ Alertas"
V_HISTORIAL, V_INDICADORES, V_ANALISIS = "🗂️ Historial", "📊 Indicadores", "📈 Análisis"
V_CONFIG, V_AYUDA = "⚙️ Configuración", "❓ Ayuda"
VIEWS = [V_TELEMETRIA, V_REPLICA, V_ALERTAS, V_HISTORIAL, V_INDICADORES, V_ANALISIS, V_CONFIG, V_AYUDA]


# Sensibilidad de cada variable a la carga operativa: a plena carga y con variación de 100 %, la lectura sube
# (o baja, en el RPM) esta fracción del tramo entre el valor base y el límite crítico.
LOAD_SENS = {"Vibración (mm/s RMS)": 0.9, "Temperatura (°C)": 0.7, "Corriente (A)": 1.0, "RPM": 0.3,
             "Error de posición (mm)": 0.0}

# Modos de falla: qué variables muestran la degradación (según las fallas del historial). El resto no se mueve.
MODO_TODAS = "Todas las variables"
MODOS_FALLA = {
    MODO_TODAS: None,
    "Solo error de posición (holgura, descalibración)": ["Error de posición (mm)"],
    "Vibración y temperatura (rodamiento, lubricación)": ["Vibración (mm/s RMS)", "Temperatura (°C)"],
    "Corriente y temperatura (servomotor)": ["Corriente (A)", "Temperatura (°C)"],
}

# Valores por defecto de los controles
DEFAULTS = {
    "data_source": SIMULACION,
    "asset_name": "CNC plasma",
    "scenario": "Falla acelerada",
    "policy": "Reactivo",
    "duration": 48,
    "seed": 42,
    "mtbf_gain": 35.0,       # supuesto con el que se obtuvieron las cifras de referencia
    "mttr_reduction": 35.0,
    "simulations": 1000,     # corridas del Monte Carlo de referencia (opciones: hasta 10,000)
    "noise_factor": 1.0,
    "missing_rate_pct": 2,
    "intervention_pct": 66,
    "detection": DETECCION_ESTADO,
    "p_prog": 70,            # historial: 10 de 14 fallas con degradación progresiva (71 %)
    "position_capture": CAPTURA_CONTINUA,
    "inspection_h": 8,
    "load_pct": 0,
    "compensate_load": False,
    "failure_mode": MODO_TODAS,
    "min_lead_h": 2.0,
}

LEVEL_COLOR = {"Normal": "#34d399", "Advertencia": "#fbbf24", "Crítico": "#fb7185", "Sin dato": "#64748b"}


# ===========================================================================
# SETTINGS
# ===========================================================================

DATA_DIR = Path(os.environ.get("GEMELO_DATA_DIR", "data"))
CONFIG_PATH = Path(os.environ.get("GEMELO_CONFIG", "gemelo_config.json"))
USERS_PATH = Path(os.environ.get("GEMELO_USERS", "users.json"))
DB_PATH = DATA_DIR / "gemelo.db"
LOG_PATH = DATA_DIR / "gemelo.log"


def get_logger() -> logging.Logger:
    """Logger de la aplicación: consola y archivo rotativo (si el directorio de datos es escribible)."""
    logger = logging.getLogger("gemelo")
    if logger.handlers:
        return logger
    logger.setLevel(os.environ.get("GEMELO_LOG_LEVEL", "INFO").upper())
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(LOG_PATH, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError:
        logger.warning("No se pudo crear el archivo de registro en %s", DATA_DIR)
    logger.propagate = False
    return logger


# ===========================================================================
# ASSETS
# ===========================================================================

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
            "Error de posición (mm)": {"base": 0.028, "noise": 0.003, "drift": 0.115, "warning": 0.10, "critical": 0.12, "direction": "high", "manual": True},
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
# (según la tabla de instrumentación).
SENSOR_INFO = {
    "CNC plasma": {
        "Vibración (mm/s RMS)": ("Acelerómetro triaxial IEPE o RS-485, IP67 (1 punto base y 1 de contingencia)", "Accionamiento del eje X (base) y carro del eje Y (contingencia)", "Gateway (MQTT u OPC UA)"),
        "Temperatura (°C)": ("PT100 clase A con transmisor aislado (2 puntos)", "Servomotores de los ejes X e Y", "Gateway (MQTT u OPC UA)"),
        "Corriente (A)": ("Transductor de núcleo partido, salida aislada", "Accionamiento principal (verificar si cubre el servomotor del eje Y)", "Gateway (MQTT u OPC UA)"),
        "Error de posición (mm)": ("Inspección del mantenimiento autónomo (TPM); el controlador F2300BT envía pulsos y no recibe realimentación de posición", "Antorcha y ejes", "Registro de inspección TPM"),
    },
    "Torno": {
        "Vibración (mm/s RMS)": ("Acelerómetro triaxial IEPE o RS-485, IP67 (1 punto base y 1 de contingencia)", "Cabezal (base) y carro longitudinal (contingencia)", "Gateway (MQTT u OPC UA)"),
        "Temperatura (°C)": ("PT100 clase A con transmisor aislado (2 puntos)", "Cabezal (caja de velocidades) y motor", "Gateway (MQTT u OPC UA)"),
        "Corriente (A)": ("Transductor de núcleo partido, salida aislada", "Accionamiento principal", "Gateway (MQTT u OPC UA)"),
        "RPM": ("Sensor inductivo u óptico, salida aislada (interpretar frente a la velocidad que selecciona el operario)", "Husillo", "Gateway (MQTT u OPC UA)"),
    },
}


# Historial de fallas (14) y clasificación propuesta; debe validarla el jefe de Mantenimiento.
# Columnas: equipo, falla registrada, ¿progresiva?, variable que la anticiparía, medio de captura.
FALLAS_HISTORIAL = [
    ("Torno", "Vibración en el cabezal por desgaste del rodamiento principal", "Sí", "Vibración y temperatura", "Sensor continuo"),
    ("CNC plasma", "Pérdida de referencia del eje X por desalineación del sensor de origen", "Parcial", "Error de posición", "Inspección TPM"),
    ("CNC plasma", "Holgura de 0,12 mm en el eje X por falta de lubricación en las guías", "Sí", "Error de posición y vibración", "Sensor e inspección"),
    ("Torno", "Holgura en la torreta portaherramientas; acabado dimensional fuera de tolerancia", "Sí", "Holgura y control dimensional", "Inspección TPM"),
    ("CNC plasma", "Desviación de la antorcha durante el corte por desajuste del eje X", "Sí", "Error de posición", "Inspección TPM"),
    ("CNC plasma", "Juego mecánico en el acoplamiento del eje Y y pérdida de precisión", "Sí", "Vibración y error de posición", "Sensor e inspección"),
    ("CNC plasma", "Sobrecarga y cortocircuito parcial en el servomotor del eje Y", "Parcial", "Corriente y temperatura", "Sensor continuo"),
    ("CNC plasma", "Falla intermitente del encoder del eje Y después de la intervención correctiva", "No", "Ninguna de las variables propuestas", "Sin cobertura"),
    ("Torno", "Desalineación de bancada de 0,15 mm/m y holgura en el carro longitudinal", "Sí", "Alineación y holgura", "Inspección TPM"),
    ("Torno", "Reaparición de holgura en el carro longitudinal por desgaste de las guías", "Sí", "Holgura y vibración", "Inspección TPM"),
    ("CNC plasma", "Descalibración del eje X por lubricación insuficiente y contaminación de las guías", "Sí", "Error de posición y vibración", "Sensor e inspección"),
    ("CNC plasma", "Pérdida de pasos del eje X por tensión inadecuada en el sistema de transmisión", "Sí", "Error de posición", "Inspección TPM"),
    ("Torno", "Desalineación del contrapunto durante el mecanizado de una pieza larga", "No", "Ninguna de las variables propuestas", "Sin cobertura"),
    ("Torno", "Recalentamiento de la caja de velocidades por lubricación insuficiente", "Sí", "Temperatura", "Sensor continuo"),
]

# Cobertura de cada falla con la instrumentación base: un acelerómetro por equipo (eje X del CNC y cabezal del torno),
# cuatro PT100, corriente, RPM del torno y las inspecciones TPM. «Parcial»: la señal principal de vibración queda en un
# punto de contingencia, pero existe otra vía (inspección). Mismo orden que FALLAS_HISTORIAL.
COBERTURA_BASE = ["Plena", "Plena", "Plena", "Plena", "Plena", "Parcial", "Plena",
                  "Sin cobertura", "Plena", "Parcial", "Plena", "Plena", "Sin cobertura", "Plena"]

# ---------------------------------------------------------------------------
# Fichas de los equipos. Las dimensiones son nominales: confirmar con la medición en planta.
# ---------------------------------------------------------------------------
EQUIPMENT_SPECS = {
    "CNC plasma": [
        ("Tipo de máquina", "Pórtico CNC de corte por plasma y oxicorte, con rieles laterales y mesa de parrilla", "Rótulo del equipo"),
        ("Fabricante (rótulo en la viga)", "«… Buluoer Intelligence»; modelo por confirmar", "Rótulo del equipo"),
        ("Controlador", "Fangling F2300BT: 2 ejes, salida de pulsos y dirección a los drivers, sin realimentación de posición; control de altura de antorcha por tensión de arco; USB y RS-232; hasta 15 m/min", "Rótulo + manual de la serie F2000"),
        ("Fuente de plasma", "LGK-200 IGBT: inversor trifásico de 380 V, clase 200 A; los catálogos difieren en corriente nominal y ciclo de trabajo", "Rótulo + catálogos de la serie LGK"),
        ("Antorchas", "Dos antorchas en el carro (plasma y oxicorte); selector Plasma/Llama en el panel", "Rótulo del equipo"),
        ("Área de corte y dimensiones", "Unos 3 m de luz de pórtico y 6 m de recorrido", "Nominal: confirmar en planta"),
        ("Fallas del historial", "8 fallas: 5 en el eje X y 3 en el eje Y", "Registro de fallas"),
        ("Datos del controlador (modelos «T»)", "Contadores de perforaciones, piezas, distancia y tiempo de corte, y tiempo de funcionamiento", "Manual de la serie F2000: verificar en la unidad"),
    ],
    "Torno": [
        ("Tipo de máquina", "Torno paralelo horizontal manual de bancada larga, con plato de garras, portaherramientas de 4 posiciones y contrapunto", "Rótulo del equipo"),
        ("Fabricante (rótulo)", "TOPTECH; modelo por confirmar", "Rótulo del equipo"),
        ("Placa de características", "Pendiente: registrar modelo, potencia, tensión y número de serie", "Por levantar en planta"),
        ("Velocidad del husillo", "Se selecciona con palancas; el panel indica límites de «≤ 500 r/min» y «≤ 600 r/min» junto a pictogramas de plato", "Rótulo del equipoo: verificar"),
        ("Fallas del historial", "6 fallas: 2 en cabezal y caja de velocidades, 3 en carro, torreta y bancada, 1 en el contrapunto", "Registro de fallas"),
        ("Dimensiones", "Unos 3.6 m de largo, 1.4 m de fondo y 1.6 m de alto", "Nominal: confirmar en planta"),
    ],
}


# Umbrales de fábrica: sirven para restaurar los valores si se editan desde la configuración.
ASSETS_FACTORY = copy.deepcopy(ASSETS)


# ===========================================================================
# THEME
# ===========================================================================

PAPER_BG = "#0b1a2a"
PLOT_BG = "#081522"
FONT_COLOR = "#b8c7d9"
GRID_COLOR = "rgba(255,255,255,.06)"
THEME = dict(paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG, font_color=FONT_COLOR)
PLOTLY_CONFIG = {"displaylogo": False}
PLOTLY_CONFIG_BARE = {"displaylogo": False, "displayModeBar": False}


def inject_css() -> None:
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


# ===========================================================================
# METRICS
# ===========================================================================

def base_metrics(asset):
    """Devuelve (horas operativas, MTBF, MTTR, disponibilidad %) del periodo diagnóstico."""
    failures = asset["failures"]
    if failures <= 0:
        raise ValueError(f"{asset.get('code', 'activo')}: se requiere al menos una falla en el historial")
    operating = asset["scheduled_hours"] - asset["downtime_hours"]
    mtbf = operating / failures
    mttr = asset["downtime_hours"] / failures
    return operating, mtbf, mttr, availability_pct(mtbf, mttr)


def availability_pct(mtbf, mttr):
    return mtbf / (mtbf + mttr) * 100


# ===========================================================================
# TELEMETRY
# ===========================================================================

@dataclass(frozen=True)
class SimConfig:
    """Parámetros de una corrida de telemetría (inmutable y hashable: sirve como llave de caché)."""

    asset_name: str
    scenario: str
    duration: float = 48
    seed: int = 42
    noise_factor: float = 1.0
    missing_rate: float = 0.0
    intervention_threshold: float = 0.66
    detection: str = DETECCION_ESTADO
    evaluate_normal: bool = False
    load_variability: float = 0.0
    compensate_load: bool = False
    inspection_h: float = 0.0
    failure_mode: str = MODO_TODAS
    thresholds_id: str = ""      # huella de los umbrales vigentes: invalida el caché al cambiarlos

    def __post_init__(self):
        if self.asset_name not in ASSETS:
            raise ValueError(f"Activo desconocido: {self.asset_name!r}")
        if self.scenario not in SCENARIOS:
            raise ValueError(f"Escenario desconocido: {self.scenario!r}")
        if self.failure_mode not in MODOS_FALLA:
            raise ValueError(f"Modo de falla desconocido: {self.failure_mode!r}")
        if self.duration <= 0:
            raise ValueError("La duración debe ser positiva")
        if not 0.0 <= self.missing_rate <= 0.5:
            raise ValueError("missing_rate debe estar entre 0 y 0.5")
        if not 0.0 <= self.load_variability <= 1.0:
            raise ValueError("load_variability debe estar entre 0 y 1")
        if self.inspection_h < 0:
            raise ValueError("inspection_h no puede ser negativo")


class Telemetry(NamedTuple):
    """Resultado de una corrida, sin pandas."""

    hours: np.ndarray
    degradation: np.ndarray              # fracción 0-1.1
    observed: dict                       # variable -> lecturas (NaN = dato perdido)
    statuses: np.ndarray                 # "Normal" | "Advertencia" | "Crítico"
    intervention_index: Optional[int]
    alerts: list                         # vacío si full=False
    condition_pct: Optional[np.ndarray]  # None si full=False
    profile: Optional[np.ndarray]        # None si full=False


def degradation_curve(points, scenario, rng):
    progress = np.linspace(0, 1, points)
    if scenario == "Operación normal":
        curve = 0.02 + 0.05 * progress
    elif scenario == "Desgaste progresivo":
        curve = 0.03 + 0.82 * np.power(progress, 1.75)
    else:
        curve = 0.04 + 1.05 / (1 + np.exp(-10 * (progress - 0.56)))
    return np.clip(curve + rng.normal(0, 0.008, points), 0, 1.10)


def learned_baseline(observed, asset, start, length):
    """Perfil normal aprendido: mediana de cada variable en una ventana de lecturas."""
    out = {}
    for variable, spec in asset["variables"].items():
        window = np.asarray(observed[variable][start:start + length], dtype=float)
        window = window[~np.isnan(window)]
        out[variable] = float(np.median(window)) if len(window) >= 3 else spec["base"]
    return out


def condition_index(observed, asset, baseline=None):
    """Índice de condición estimado SOLO con lecturas: mayor desviación normalizada
    (0 = perfil normal, 1 = límite crítico) entre las variables, suavizada con mediana de 3.
    Sin `baseline` usa las constantes del diagnóstico; con `baseline`, el perfil normal aprendido."""
    parts = []
    for variable, spec in asset["variables"].items():
        values = np.asarray(observed[variable], dtype=float)
        base = spec["base"] if baseline is None else baseline[variable]
        if spec["direction"] == "high":
            parts.append((values - base) / max(spec["critical"] - base, 1e-9))
        else:
            parts.append((base - values) / max(base - spec["critical"], 1e-9))
    raw = np.fmax.reduce(np.vstack(parts), axis=0)  # ignora NaN, igual que el máximo de pandas
    return pd.Series(raw).rolling(3, min_periods=1).median().to_numpy()


def persistent_hit(above, intervals=PERSISTENCE_INTERVALS):
    """Primer instante en que la condición se mantiene durante `intervals` lecturas seguidas."""
    run = pd.Series(above.astype(int)).rolling(intervals).sum().to_numpy()
    hits = np.where(run >= intervals)[0]
    return int(hits[0]) if len(hits) else None


NIVEL_INTERVENCION_EJECUTADA = "Intervención ejecutada"
NIVEL_INTERVENCION_PROPUESTA = "Intervención propuesta"


def is_intervention(level) -> bool:
    return str(level).startswith("Intervención")


def classify_readings(hours, observed_all, asset, full=True):
    """Estado de cada instante (peor nivel entre las variables) y primera alerta de cada variable."""
    statuses = np.full(len(hours), "Normal", dtype=object)
    alerts = []
    for variable, spec in asset["variables"].items():
        observed = np.asarray(observed_all[variable], dtype=float)
        with np.errstate(invalid="ignore"):
            if spec["direction"] == "high":
                warning_mask = observed >= spec["warning"]
                critical_mask = observed >= spec["critical"]
            else:
                warning_mask = observed <= spec["warning"]
                critical_mask = observed <= spec["critical"]
        statuses[warning_mask] = "Advertencia"
        statuses[critical_mask] = "Crítico"
        if full:
            warning_positions = np.where(warning_mask)[0]
            if len(warning_positions):
                idx = int(warning_positions[0])
                alerts.append({
                    "Hora": round(float(hours[idx]), 2),
                    "Variable": variable,
                    "Lectura": round(float(observed[idx]), 3),
                    "Umbral": spec["warning"],
                    "Nivel": "Crítico" if critical_mask[idx] else "Advertencia",
                })
    return statuses, alerts


def run_simulation(sim: SimConfig, policy: str, full: bool = True) -> Telemetry:
    """Genera la telemetría simulada de un equipo.

    policy: "Reactivo" (sin intervención) o "Predictivo" (el gemelo propone intervenir).
    full: si es False se omiten alertas, índice de condición y perfil (suficiente para los análisis
        que solo necesitan los estados y la hora de intervención).
    load_variability: fracción 0-1 de variación de la carga operativa (0 = sin variación).
    compensate_load: el gemelo descuenta el régimen de carga (perfil normal por régimen); el régimen se conoce
        de forma imperfecta, por lo que queda 20 % del efecto.
    inspection_h: horas entre inspecciones TPM para las variables manuales (0 = captura continua cada 5 min).
    failure_mode: variables que muestran la degradación (MODOS_FALLA); por defecto, todas.
    """
    if policy not in POLICIES:
        raise ValueError(f"Política desconocida: {policy!r}")
    asset_name, scenario, duration, seed = sim.asset_name, sim.scenario, sim.duration, sim.seed
    noise_factor, missing_rate = sim.noise_factor, sim.missing_rate
    intervention_threshold, detection = sim.intervention_threshold, sim.detection
    evaluate_normal, load_variability = sim.evaluate_normal, sim.load_variability
    compensate_load, inspection_h, failure_mode = sim.compensate_load, sim.inspection_h, sim.failure_mode
    asset = ASSETS[asset_name]
    rng = np.random.default_rng(seed)
    points = int(duration * SAMPLES_PER_HOUR) + 1
    hours = np.linspace(0, duration, points)
    degradation = degradation_curve(points, scenario, rng)

    # El ruido y las pérdidas de datos se sortean en el mismo orden que la versión original.
    noise, missing = {}, {}
    for variable, spec in asset["variables"].items():
        noise[variable] = rng.normal(0, spec["noise"] * noise_factor, points)
        missing[variable] = rng.random(points) < missing_rate

    # Variación de carga operativa, con semilla propia para no alterar las demás señales.
    load_effect = {variable: 0.0 for variable in asset["variables"]}
    if load_variability > 0:
        rng_load = np.random.default_rng(seed + 7)
        load = np.repeat(rng_load.uniform(0, 1, int(np.ceil(points / SAMPLES_PER_HOUR)) + 1), SAMPLES_PER_HOUR)[:points]
        thermal = pd.Series(load).rolling(THERMAL_INERTIA_SAMPLES, min_periods=1).mean().to_numpy()  # inercia térmica
        for variable, spec in asset["variables"].items():
            sens = LOAD_SENS.get(variable, 0.0)
            span = abs(spec["critical"] - spec["base"])
            sign = 1.0 if spec["drift"] >= 0 else -1.0
            series = thermal if "Temperatura" in variable else load
            load_effect[variable] = load_variability * sens * span * sign * series
    inspection_step = max(1, int(round(inspection_h * SAMPLES_PER_HOUR))) if inspection_h > 0 else 0
    affected = MODOS_FALLA.get(failure_mode)
    weight = {v: (1.0 if affected is None or v in affected else 0.0) for v in asset["variables"]}

    def observe(deg, compensated=False):
        out = {}
        for variable, spec in asset["variables"].items():
            effect = load_effect[variable] * (0.2 if compensated else 1.0)
            values = np.array(spec["base"] + deg * spec["drift"] * weight[variable] + noise[variable] + effect, dtype=float)
            values[missing[variable]] = np.nan
            if inspection_step and spec.get("manual"):
                keep = np.zeros(points, dtype=bool)
                keep[::inspection_step] = True      # la variable solo existe en cada inspección
                values[~keep] = np.nan
            out[variable] = values
        return out

    def held(obs):
        """Para las variables de inspección, el gemelo conserva el último valor medido hasta la siguiente inspección."""
        out = dict(obs)
        for variable, spec in asset["variables"].items():
            if inspection_step and spec.get("manual"):
                out[variable] = pd.Series(obs[variable]).ffill().to_numpy()
        return out

    learn_samples = int(LEARN_H * SAMPLES_PER_HOUR)
    intervention_index = None
    if policy == "Predictivo" and (scenario != "Operación normal" or evaluate_normal):
        if detection == DETECCION_ESTADO:
            # Estado interno: cruce del umbral por el estado simulado + intervalos de persistencia.
            candidates = np.where(degradation >= intervention_threshold)[0]
            if len(candidates):
                intervention_index = int(min(candidates[0] + PERSISTENCE_INTERVALS, points - 1))
        else:
            # Detección con lecturas: perfil normal aprendido y persistencia real de 4 lecturas seguidas.
            detect_obs = held(observe(degradation, compensated=compensate_load))
            baseline = learned_baseline(detect_obs, asset, 0, learn_samples)
            estimate = condition_index(detect_obs, asset, baseline)
            above = np.nan_to_num(estimate, nan=0.0) >= intervention_threshold
            above[:learn_samples] = False           # mientras aprende el perfil normal no propone intervenir
            intervention_index = persistent_hit(above, PERSISTENCE_INTERVALS)
        if intervention_index is not None:
            residual = degradation[intervention_index]
            degradation[intervention_index:] = np.maximum(
                0.06,
                residual * 0.08 + np.linspace(0, 0.10, points - intervention_index),
            )

    observed_all = observe(degradation)
    statuses, alerts = classify_readings(hours, observed_all, asset, full)

    if intervention_index is not None:
        statuses[intervention_index:] = "Normal"
        if full:
            alerts.append({
                "Hora": round(float(hours[intervention_index]), 2),
                "Variable": "Mantenimiento",
                "Lectura": 0.0,
                "Umbral": 0.0,
                "Nivel": NIVEL_INTERVENCION_EJECUTADA,
            })

    if not full:
        return Telemetry(hours, degradation, observed_all, statuses, intervention_index, [], None, None)

    # Índice de condición mostrado: constantes del diagnóstico o perfil aprendido y recalibrado.
    if detection == DETECCION_ESTADO:
        index_pct = condition_index(observed_all, asset) * 100
        profile = np.full(points, "Constantes del diagnóstico", dtype=object)
    else:
        shown = held(observe(degradation, compensated=compensate_load))
        index_pct = condition_index(shown, asset, learned_baseline(shown, asset, 0, learn_samples)) * 100
        profile = np.full(points, "Aprendido (inicial)", dtype=object)
        if intervention_index is not None and intervention_index + 1 + learn_samples <= points:
            start = intervention_index + 1
            recal_from = start + learn_samples
            index_pct[recal_from:] = (condition_index(shown, asset, learned_baseline(shown, asset, start, learn_samples)) * 100)[recal_from:]
            profile[recal_from:] = "Recalibrado tras la intervención"
    return Telemetry(hours, degradation, observed_all, statuses, intervention_index, alerts, index_pct, profile)


def to_frames(t: Telemetry):
    """Convierte una corrida completa en (telemetría, alertas, índice de intervención) para la interfaz."""
    columns = {"Hora simulada": t.hours, "Degradación": t.degradation * 100}
    columns.update(t.observed)
    columns["Estado"] = t.statuses
    columns["Índice de condición estimado (%)"] = t.condition_pct
    columns["Perfil normal"] = t.profile
    df = pd.DataFrame(columns)
    alerts_df = pd.DataFrame(t.alerts).sort_values("Hora") if t.alerts else pd.DataFrame()
    return df, alerts_df, t.intervention_index


def simulate_frames(sim: SimConfig, policy: str):
    return to_frames(run_simulation(sim, policy, full=True))


def data_quality(df, asset_name, exclude_manual=False):
    """Completitud de las lecturas de sensores; las variables de inspección manual se excluyen si se piden así."""
    variables = [v for v, spec in ASSETS[asset_name]["variables"].items() if not (exclude_manual and spec.get("manual"))]
    expected = len(df) * len(variables)
    observed = int(df[variables].notna().sum().sum())
    return observed / expected * 100 if expected else 0.0


# ===========================================================================
# THRESHOLDS
# ===========================================================================

EDITABLE = ("warning", "critical")


def validate_thresholds(direction: str, base: float, warning: float, critical: float) -> list:
    """Devuelve la lista de errores (vacía si los umbrales son coherentes con la dirección de la degradación)."""
    errors = []
    if direction == "high":
        if not base < warning < critical:
            errors.append("Debe cumplirse: valor normal < advertencia < crítico.")
    else:
        if not base > warning > critical:
            errors.append("Debe cumplirse: valor normal > advertencia > crítico.")
    return errors


def load_overrides(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"El archivo de umbrales {path} no es un JSON válido: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("El archivo de umbrales debe ser un objeto JSON {equipo: {variable: {...}}}.")
    return data


def apply_overrides(assets: dict, overrides: dict) -> list:
    """Aplica los umbrales del archivo a `assets` (en el sitio). Devuelve mensajes sobre lo aplicado o ignorado."""
    messages = []
    for asset_name, variables in overrides.items():
        if asset_name not in assets:
            messages.append(f"Equipo desconocido ignorado: {asset_name}")
            continue
        for variable, values in variables.items():
            spec = assets[asset_name]["variables"].get(variable)
            if spec is None:
                messages.append(f"Variable desconocida ignorada: {asset_name} / {variable}")
                continue
            new = {k: float(values[k]) for k in EDITABLE if k in values}
            warning, critical = new.get("warning", spec["warning"]), new.get("critical", spec["critical"])
            errors = validate_thresholds(spec["direction"], spec["base"], warning, critical)
            if errors:
                messages.append(f"{asset_name} / {variable}: {errors[0]} Se conservan los valores por defecto.")
                continue
            spec.update(new)
            messages.append(f"{asset_name} / {variable}: advertencia {warning}, crítico {critical}")
    return messages


def save_overrides(path: Path, overrides: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(overrides, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def fingerprint(assets: dict) -> str:
    """Huella corta de los umbrales vigentes: sirve como parte de la llave de caché."""
    payload = {a: {v: [s["warning"], s["critical"]] for v, s in spec["variables"].items()} for a, spec in assets.items()}
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:10]


def snapshot_defaults(assets: dict) -> dict:
    return copy.deepcopy(assets)


def reload_thresholds(assets: dict, factory: dict, path: Path) -> list:
    """Vuelve a los umbrales de fábrica y aplica encima los del archivo. Se llama en cada ejecución (es barato e idempotente)."""
    for asset_name, spec in factory.items():
        for variable, values in spec["variables"].items():
            assets[asset_name]["variables"][variable].update({k: values[k] for k in EDITABLE})
    return apply_overrides(assets, load_overrides(path))


# ===========================================================================
# DETECTION
# ===========================================================================

DEFAULT_TRAJECTORIES = 150


def failing_scenario(sim: SimConfig) -> str:
    return sim.scenario if sim.scenario != "Operación normal" else "Desgaste progresivo"


def reference_config(sim: SimConfig) -> SimConfig:
    """Configuración canónica de la referencia reactiva: solo conserva lo que la afecta."""
    return SimConfig(
        asset_name=sim.asset_name, scenario=failing_scenario(sim), duration=sim.duration, seed=0,
        noise_factor=sim.noise_factor, missing_rate=sim.missing_rate, failure_mode=sim.failure_mode,
        thresholds_id=sim.thresholds_id,
    )


def detection_config(sim: SimConfig) -> SimConfig:
    """Configuración canónica de la corrida predictiva con trayectoria de falla (la semilla se fija por trayectoria)."""
    return replace(sim, scenario=failing_scenario(sim), seed=0, evaluate_normal=False)


def critical_hours(ref: SimConfig, n: int = DEFAULT_TRAJECTORIES) -> np.ndarray:
    """Hora de condición crítica sin intervenir; NaN si la trayectoria no llega a ella en el horizonte."""
    out = np.full(n, np.nan)
    for k in range(1, n + 1):
        t = run_simulation(replace(ref, seed=SEED_FAILING + k), "Reactivo", full=False)
        critical = np.flatnonzero(t.statuses == "Crítico")
        if len(critical):
            out[k - 1] = t.hours[critical[0]]
    return out


def intervention_hours(sim: SimConfig, critical: np.ndarray) -> np.ndarray:
    """Hora de intervención propuesta en las trayectorias que llegan a condición crítica (NaN si no hay detección)."""
    out = np.full(len(critical), np.nan)
    for k in np.flatnonzero(~np.isnan(critical)):
        t = run_simulation(replace(sim, seed=SEED_FAILING + int(k) + 1), "Predictivo", full=False)
        if t.intervention_index is not None:
            out[k] = t.hours[t.intervention_index]
    return out


def false_alarm_rate(sim: SimConfig, n: int = DEFAULT_TRAJECTORIES) -> float:
    """Fracción de corridas de operación normal en las que el gemelo propone intervenir."""
    normal = replace(sim, scenario="Operación normal", evaluate_normal=True)
    hits = 0
    for k in range(1, n + 1):
        t = run_simulation(replace(normal, seed=SEED_NORMAL + k), "Predictivo", full=False)
        hits += t.intervention_index is not None
    return hits / n


def summarize_detection(critical: np.ndarray, interventions: np.ndarray, false_alarms: float, min_lead_h: float) -> dict:
    has_failure = ~np.isnan(critical)
    leads = critical[has_failure] - interventions[has_failure]   # NaN = falla sin detección
    valid = leads[~np.isnan(leads)]
    return {
        "fallas": int(has_failure.sum()),
        "p_anticipada": float(np.mean(leads >= min_lead_h)) if len(leads) else float("nan"),
        "p_detectada": float(np.mean(~np.isnan(leads))) if len(leads) else float("nan"),
        "aviso_mediano_h": float(np.median(valid)) if len(valid) else float("nan"),
        "aviso_p10_h": float(np.percentile(valid, 10)) if len(valid) else float("nan"),
        "falsas_alarmas": false_alarms,
    }


# ===========================================================================
# EXPERIMENTS
# ===========================================================================

def experiment_matrix(sim: SimConfig, simulate: Callable = simulate_frames) -> pd.DataFrame:
    rows = []
    for scenario_name in SCENARIOS:
        for policy_name in POLICIES:
            run_cfg = replace(sim, scenario=scenario_name, seed=sim.seed + len(rows) * SEED_STEP_EXPERIMENT)
            trial, trial_alerts, trial_intervention = simulate(run_cfg, policy_name)
            critical = trial.index[trial["Estado"] == "Crítico"].tolist()
            predictive_alerts = (
                trial_alerts[~trial_alerts["Nivel"].map(is_intervention)]
                if not trial_alerts.empty and "Nivel" in trial_alerts.columns
                else pd.DataFrame()
            )
            first_warning = float(predictive_alerts.iloc[0]["Hora"]) if not predictive_alerts.empty else np.nan
            critical_hour = float(trial.loc[critical[0], "Hora simulada"]) if critical else np.nan
            rows.append({
                "Escenario": scenario_name,
                "Política": policy_name,
                "Calidad de datos (%)": round(data_quality(trial, sim.asset_name, sim.inspection_h > 0), 2),
                "Primera alerta (h)": round(first_warning, 2) if not np.isnan(first_warning) else np.nan,
                "Condición crítica (h)": round(critical_hour, 2) if not np.isnan(critical_hour) else np.nan,
                "Intervención (h)": round(float(trial.loc[trial_intervention, "Hora simulada"]), 2)
                if trial_intervention is not None else np.nan,
                "Salud final (%)": round(max(0.0, 100 - float(trial.iloc[-1]["Degradación"])), 2),
            })
    return pd.DataFrame(rows)


# ===========================================================================
# MONTE_CARLO
# ===========================================================================

def monte_carlo_sample(asset_name, mtbf_gain, mttr_reduction, simulations, seed):
    if not 0 <= mttr_reduction < 100:
        raise ValueError("mttr_reduction debe estar entre 0 y 100")
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


def seed_means(asset_name, mtbf_gain, mttr_reduction, simulations, n_seeds=100) -> pd.DataFrame:
    """Repite el Monte Carlo con otras semillas y conserva solo el promedio predictivo de cada una."""
    rows = []
    for s in range(1, n_seeds + 1):
        sample = monte_carlo_sample(asset_name, mtbf_gain, mttr_reduction, simulations, s)
        pred = sample[sample["Política"] == "Predictiva"]
        rows.append({
            "Semilla": s,
            "MTBF (h)": pred["MTBF (h)"].mean(),
            "MTTR (h)": pred["MTTR (h)"].mean(),
            "Disponibilidad (%)": pred["Disponibilidad (%)"].mean(),
        })
    return pd.DataFrame(rows)


# ===========================================================================
# INGEST
# ===========================================================================

TIME_NAMES = {"timestamp", "fecha", "fechahora", "fecha_hora", "hora", "time", "datetime", "tiempo"}
MAX_POINTS = 20_000                      # ~69 días a una lectura cada 5 min
MIN_HOURS = LEARN_H + 1                  # el perfil normal se aprende con las primeras horas


class IngestError(ValueError):
    """El archivo no se puede interpretar; el mensaje está pensado para mostrarse al usuario."""


def _normalize(name: str) -> str:
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii").lower().strip()
    return text.split("(")[0].strip().replace("_", " ")


def _to_number(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        series = series.astype(str).str.strip().str.replace(",", ".", regex=False)
    return pd.to_numeric(series, errors="coerce")


def parse_csv(data: bytes, asset_name: str) -> pd.DataFrame:
    """Lee el CSV y devuelve una lectura cada 5 min (columnas: 'Horas' y una por variable; vacío = dato perdido)."""
    if asset_name not in ASSETS:
        raise IngestError(f"Equipo desconocido: {asset_name}")
    if not data:
        raise IngestError("El archivo está vacío.")
    try:
        raw = pd.read_csv(io.BytesIO(data), sep=None, engine="python")
    except Exception as exc:  # noqa: BLE001
        raise IngestError(f"No se pudo leer el CSV: {exc}") from exc
    if raw.shape[1] < 2 or raw.empty:
        raise IngestError("El CSV debe tener una columna de tiempo y una columna por variable.")

    lookup = {_normalize(c): c for c in raw.columns}
    time_col = next((lookup[k] for k in lookup if k in TIME_NAMES), raw.columns[0])

    columns = {}
    missing = []
    for variable in ASSETS[asset_name]["variables"]:
        source = lookup.get(_normalize(variable))
        if source is None or source == time_col:
            missing.append(variable)
        else:
            columns[variable] = source
    if missing:
        raise IngestError(
            f"Faltan columnas para {asset_name}: {', '.join(missing)}. "
            f"Columnas encontradas: {', '.join(map(str, raw.columns))}."
        )

    if pd.api.types.is_numeric_dtype(raw[time_col]):
        hours = raw[time_col].astype(float).to_numpy()
    else:
        stamps = pd.to_datetime(raw[time_col], errors="coerce", format="ISO8601")      # 2026-01-31 08:05:00
        if stamps.isna().mean() > 0.01:
            stamps = pd.to_datetime(raw[time_col], errors="coerce", dayfirst=True)     # 31/01/2026 08:05
        if stamps.isna().mean() > 0.01:
            raise IngestError(f"La columna de tiempo '{time_col}' tiene fechas que no se pueden interpretar.")
        hours = ((stamps - stamps.min()).dt.total_seconds() / 3600.0).to_numpy()

    frame = pd.DataFrame({"Horas": hours, **{v: _to_number(raw[c]) for v, c in columns.items()}})
    frame = frame[np.isfinite(frame["Horas"])]
    if frame.empty:
        raise IngestError("No hay filas con tiempo válido.")
    frame["Horas"] -= frame["Horas"].min()

    bucket = np.floor(frame["Horas"] * SAMPLES_PER_HOUR + 1e-9).astype(int)
    if bucket.max() + 1 > MAX_POINTS:
        raise IngestError(f"El archivo cubre demasiado tiempo (máximo {MAX_POINTS // SAMPLES_PER_HOUR} h). Cárgalo por tramos.")
    grid = frame.drop(columns="Horas").groupby(bucket).mean().reindex(range(int(bucket.max()) + 1))
    grid.insert(0, "Horas", grid.index.to_numpy() / SAMPLES_PER_HOUR)
    grid = grid.reset_index(drop=True)
    if grid["Horas"].iloc[-1] < MIN_HOURS:
        raise IngestError(f"Se necesitan al menos {MIN_HOURS:g} h de datos: las primeras {LEARN_H:g} h se usan para aprender el perfil normal.")
    return grid


def analyze_readings(grid: pd.DataFrame, asset_name: str, intervention_threshold: float):
    """Mismo resultado que `simulate_frames` (telemetría, alertas, índice de intervención) a partir de lecturas reales."""
    asset = ASSETS[asset_name]
    hours = grid["Horas"].to_numpy()
    observed = {v: grid[v].to_numpy(dtype=float) for v in asset["variables"]}
    statuses, alerts = classify_readings(hours, observed, asset, full=True)

    learn_samples = int(LEARN_H * SAMPLES_PER_HOUR)
    baseline = learned_baseline(observed, asset, 0, learn_samples)
    estimate = condition_index(observed, asset, baseline)
    above = np.nan_to_num(estimate, nan=0.0) >= intervention_threshold
    above[:learn_samples] = False                    # mientras aprende el perfil normal no propone intervenir
    index = persistent_hit(above)
    if index is not None:
        alerts.append({
            "Hora": round(float(hours[index]), 2), "Variable": "Mantenimiento", "Lectura": 0.0, "Umbral": 0.0,
            "Nivel": NIVEL_INTERVENCION_PROPUESTA,
        })
    index_pct = estimate * 100
    columns = {"Hora simulada": hours, "Degradación": np.clip(np.nan_to_num(index_pct, nan=0.0), 0, 100)}
    columns.update(observed)
    columns["Estado"] = statuses
    columns["Índice de condición estimado (%)"] = index_pct
    columns["Perfil normal"] = np.full(len(hours), "Aprendido (inicial)", dtype=object)
    df = pd.DataFrame(columns)
    alerts_df = pd.DataFrame(alerts).sort_values("Hora") if alerts else pd.DataFrame()
    return df, alerts_df, index


def template_csv(asset_name: str) -> bytes:
    """CSV de ejemplo con el formato esperado (48 h con desgaste progresivo, una lectura cada 5 min)."""
    t = run_simulation(SimConfig(asset_name, "Desgaste progresivo", 48, 7), "Reactivo", full=False)
    frame = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=len(t.hours), freq="5min")})
    for variable, values in t.observed.items():
        frame[variable] = np.round(values, 3)
    return frame.to_csv(index=False).encode("utf-8")


# ===========================================================================
# AUTH
# ===========================================================================

ITERATIONS = 200_000
ROLES = {
    "operador": {"ver"},
    "jefe_mantenimiento": {"ver", "decidir"},
    "admin": {"ver", "decidir", "configurar"},
}
ROLE_LABEL = {"operador": "Operador", "jefe_mantenimiento": "Jefe de Mantenimiento", "admin": "Administrador"}


def hash_password(password: str, salt: bytes = None) -> dict:
    salt = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, ITERATIONS)
    return {"salt": salt.hex(), "hash": digest.hex(), "iterations": ITERATIONS}


def verify_password(password: str, record: dict) -> bool:
    salt = bytes.fromhex(record["salt"])
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, int(record.get("iterations", ITERATIONS)))
    return hmac.compare_digest(digest.hex(), record["hash"])


def load_users(path: Path) -> dict:
    """Usuarios del archivo JSON. Sin archivo => {} (modo demostración sin autenticación)."""
    path = Path(path)
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    users = data.get("users", {})
    for name, record in users.items():
        if record.get("role") not in ROLES:
            raise ValueError(f"Usuario {name}: rol inválido {record.get('role')!r}. Opciones: {', '.join(ROLES)}")
    return users


def authenticate(users: dict, username: str, password: str):
    """Devuelve {'username', 'name', 'role'} si las credenciales son correctas; None si no."""
    record = users.get(username.strip())
    # Se calcula el hash aun si el usuario no existe, para no revelar por tiempo qué usuarios existen.
    dummy = {"salt": "00" * 16, "hash": "00" * 32, "iterations": ITERATIONS}
    ok = verify_password(password, record if record else dummy)
    if record and ok:
        return {"username": username.strip(), "name": record.get("name", username.strip()), "role": record["role"]}
    return None


def can(user: dict, permission: str) -> bool:
    return permission in ROLES.get(user["role"], set())


DEMO_USER = {"username": "demo", "name": "Modo demostración", "role": "admin"}


# ===========================================================================
# STORAGE
# ===========================================================================

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    source TEXT NOT NULL,
    asset TEXT NOT NULL,
    event_type TEXT NOT NULL,
    hour REAL,
    variable TEXT,
    level TEXT,
    detail TEXT,
    ref TEXT UNIQUE
);
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    ot_id TEXT NOT NULL,
    asset TEXT NOT NULL,
    decision TEXT NOT NULL,
    username TEXT NOT NULL,
    source TEXT NOT NULL,
    note TEXT
);
CREATE INDEX IF NOT EXISTS idx_decisions_ot ON decisions (ot_id, id);
"""
VALID_DECISIONS = ("Aprobar", "Rechazar")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


@contextmanager
def connect(db_path: Path):
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=10)
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def log_event(db_path, source, asset, event_type, hour=None, variable=None, level=None, detail=None, ref=None) -> bool:
    """Registra un evento. Si `ref` ya existe no duplica y devuelve False (sirve para notificar una sola vez)."""
    with connect(db_path) as conn:
        cur = conn.execute(
            "INSERT OR IGNORE INTO events (ts, source, asset, event_type, hour, variable, level, detail, ref) VALUES (?,?,?,?,?,?,?,?,?)",
            (_now(), source, asset, event_type, hour, variable, level, detail, ref),
        )
        return cur.rowcount == 1


def record_decision(db_path, ot_id, asset, decision, username, source, note="") -> None:
    if decision not in VALID_DECISIONS:
        raise ValueError(f"Decisión inválida: {decision!r}")
    with connect(db_path) as conn:
        conn.execute(
            "INSERT INTO decisions (ts, ot_id, asset, decision, username, source, note) VALUES (?,?,?,?,?,?,?)",
            (_now(), ot_id, asset, decision, username, source, note),
        )


def last_decision(db_path, ot_id):
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT ts, decision, username FROM decisions WHERE ot_id = ? ORDER BY id DESC LIMIT 1", (ot_id,)
        ).fetchone()
    return {"ts": row[0], "decision": row[1], "username": row[2]} if row else None


def list_events(db_path, limit=500) -> pd.DataFrame:
    with connect(db_path) as conn:
        return pd.read_sql_query(
            "SELECT ts AS Fecha, source AS Fuente, asset AS Equipo, event_type AS Evento, hour AS Hora, variable AS Variable, level AS Nivel, detail AS Detalle "
            "FROM events ORDER BY id DESC LIMIT ?", conn, params=(int(limit),))


def list_decisions(db_path, limit=500) -> pd.DataFrame:
    with connect(db_path) as conn:
        return pd.read_sql_query(
            "SELECT ts AS Fecha, ot_id AS Orden, asset AS Equipo, decision AS Decisión, username AS Usuario, source AS Fuente, note AS Nota "
            "FROM decisions ORDER BY id DESC LIMIT ?", conn, params=(int(limit),))


# ===========================================================================
# NOTIFY
# ===========================================================================

_log = get_logger()


def channels_configured() -> dict:
    return {
        "correo": bool(os.environ.get("GEMELO_SMTP_HOST") and os.environ.get("GEMELO_MAIL_TO")),
        "webhook": bool(os.environ.get("GEMELO_WEBHOOK_URL")),
    }


def _send_mail(subject: str, body: str) -> None:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = os.environ.get("GEMELO_MAIL_FROM", os.environ.get("GEMELO_SMTP_USER", "gemelo@localhost"))
    msg["To"] = [a.strip() for a in os.environ["GEMELO_MAIL_TO"].split(",") if a.strip()]
    msg.set_content(body)
    port = int(os.environ.get("GEMELO_SMTP_PORT", "587"))
    with smtplib.SMTP(os.environ["GEMELO_SMTP_HOST"], port, timeout=15) as server:
        if os.environ.get("GEMELO_SMTP_TLS", "1") != "0":
            server.starttls()
        if os.environ.get("GEMELO_SMTP_USER"):
            server.login(os.environ["GEMELO_SMTP_USER"], os.environ.get("GEMELO_SMTP_PASSWORD", ""))
        server.send_message(msg)


def _send_webhook(text: str) -> None:
    url = os.environ["GEMELO_WEBHOOK_URL"]
    if not url.lower().startswith(("https://", "http://")):
        raise ValueError("GEMELO_WEBHOOK_URL debe empezar con http:// o https://")
    request = urllib.request.Request(url, data=json.dumps({"text": text}).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=10):  # noqa: S310  (esquema validado arriba)
        pass


def notify(subject: str, body: str) -> list:
    """Envía por los canales configurados. Devuelve los que funcionaron; los errores se registran, no se propagan."""
    sent = []
    configured = channels_configured()
    if configured["correo"]:
        try:
            _send_mail(subject, body)
            sent.append("correo")
        except Exception:  # noqa: BLE001  (la caída de un canal no debe tumbar la aplicación)
            _log.exception("Fallo al enviar el correo de notificación")
    if configured["webhook"]:
        try:
            _send_webhook(f"{subject}\n{body}")
            sent.append("webhook")
        except Exception:  # noqa: BLE001
            _log.exception("Fallo al enviar el webhook de notificación")
    return sent


# ===========================================================================
# EVENTS
# ===========================================================================

_log = get_logger()


def data_fingerprint(csv_bytes: bytes) -> str:
    return hashlib.sha1(csv_bytes).hexdigest()[:8]


def register_proposal(asset_name, data_id, df, alerts, intervention_index) -> bool:
    """Guarda en el historial la intervención propuesta y avisa una sola vez (por equipo y archivo de datos)."""
    if intervention_index is None:
        return False
    hour = float(df.iloc[intervention_index]["Hora simulada"])
    ref = f"{asset_name}|{data_id}|{intervention_index}"
    variable = _trigger_variable(alerts)
    detail = f"Intervención propuesta a las {hour:.2f} h del archivo de datos {data_id}."
    is_new = log_event(DB_PATH, "datos reales", asset_name, "Intervención propuesta", hour, variable, "Intervención propuesta", detail, ref)
    if is_new:
        sent = notify(f"Gemelo digital: intervención propuesta en {asset_name}", detail + " Revisa la vista Alertas para aprobar o rechazar la orden.")
        _log.info("Intervención propuesta en %s (hora %.2f). Notificada por: %s", asset_name, hour, ", ".join(sent) or "ningún canal configurado")
    return is_new


def _trigger_variable(alerts):
    if alerts is None or alerts.empty:
        return None
    first = alerts[alerts["Variable"] != "Mantenimiento"]
    return str(first.iloc[0]["Variable"]) if not first.empty else None


# ===========================================================================
# READINGS
# ===========================================================================

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



def verdict(ok):
    return "Cumple" if ok else "No cumple"


# ===========================================================================
# CARDS
# ===========================================================================

def status_card(asset_name, status, health, hour, live=False):
    css_status = {"Normal": "normal", "Advertencia": "warning", "Crítico": "critical"}[status]
    label = {"Normal": "OPERACIÓN NORMAL", "Advertencia": "REQUIERE ATENCIÓN", "Crítico": "RIESGO DE FALLA"}[status]
    asset = ASSETS[asset_name]
    twin = "Gemelo digital" if live else "Gemelo simulado"
    health_word = "estimado" if live else "simulado"
    st.markdown(
        f'<div class="status-card status-{css_status}">'
        f'<div class="machine-name">{asset_name} · {asset["code"]}</div>'
        f'<div class="machine-meta">Área de {asset["area"]} · {twin} · hora {hour:.2f}</div>'
        f'<span class="status-pill pill-{css_status}">{label}</span>'
        f'<div class="small-muted" style="margin-top:12px">Índice de salud {health_word}: '
        f'<b style="color:#f8fafc">{health:.1f}%</b></div></div>',
        unsafe_allow_html=True,
    )


def reading_card(variable, value, cfg, age_steps, inspected=False):
    level = reading_level(value, cfg)
    color = LEVEL_COLOR[level]
    note = ""
    if inspected and cfg.get("manual"):
        when = f"hace {age_steps * 5 / 60:.1f} h" if age_steps else "en este instante"
        note = f'<div class="reading-meta">Última inspección TPM {when}</div>'
    elif age_steps:
        note = f'<div class="reading-meta">Dato perdido: última lectura válida hace {age_steps * 5} min</div>'
    st.markdown(
        f'<div class="reading" style="border-left:4px solid {color}">'
        f'<div class="reading-label">{variable}</div>'
        f'<div class="reading-value">{value_text(variable, value)}</div>'
        f'<div class="reading-meta"><b style="color:{color}">{level}</b> · advertencia {cfg["warning"]} · crítico {cfg["critical"]}</div>'
        f'{note}</div>',
        unsafe_allow_html=True,
    )


# ===========================================================================
# CHARTS
# ===========================================================================

def telemetry_figure(df, asset_name, hour=None, inspected=False):
    variables = list(ASSETS[asset_name]["variables"].keys())
    colors = ["#22d3ee", "#fbbf24", "#a78bfa", "#34d399"]
    fig = make_subplots(rows=2, cols=2, subplot_titles=variables, vertical_spacing=0.17)
    for index, variable in enumerate(variables):
        row, col = index // 2 + 1, index % 2 + 1
        cfg = ASSETS[asset_name]["variables"][variable]
        sparse = inspected and cfg.get("manual")
        fig.add_trace(
            go.Scatter(x=df["Hora simulada"], y=df[variable], mode="lines+markers" if sparse else "lines", name=variable,
                       connectgaps=bool(sparse), marker=dict(size=7), line=dict(color=colors[index], width=2.4), showlegend=False),
            row=row, col=col,
        )
        fig.add_hline(y=cfg["warning"], line_dash="dash", line_color="#fbbf24", opacity=.75, row=row, col=col)
        fig.add_hline(y=cfg["critical"], line_dash="dot", line_color="#fb7185", opacity=.80, row=row, col=col)
        if hour is not None:
            fig.add_vline(x=hour, line_color="#cbd5e1", line_width=1, opacity=.55, row=row, col=col)
    fig.update_layout(height=560, margin=dict(l=20, r=20, t=55, b=20), paper_bgcolor="#0b1a2a",
                      plot_bgcolor="#081522", font_color="#b8c7d9", hovermode="x unified")
    fig.update_xaxes(title_text="Horas", gridcolor=GRID_COLOR)
    fig.update_yaxes(gridcolor=GRID_COLOR)
    return fig


def timeline_figure(df_re, df_pr, alerts_re, alerts_pr, idx_pr, hour):
    """Franja de estado en el tiempo para las dos políticas: muestra cuándo pasa a amarillo y a rojo."""
    code = {"Normal": 0, "Advertencia": 1, "Crítico": 2}
    hours = df_re["Hora simulada"].to_numpy()
    z = [df_re["Estado"].map(code).to_numpy(), df_pr["Estado"].map(code).to_numpy()]
    text = [df_re["Estado"].to_numpy(), df_pr["Estado"].to_numpy()]
    fig = go.Figure(go.Heatmap(
        z=z, x=hours, y=["Reactiva", "Predictiva"], zmin=0, zmax=2, text=text, showscale=False,
        colorscale=[[0, LEVEL_COLOR["Normal"]], [0.3333, LEVEL_COLOR["Normal"]], [0.3334, LEVEL_COLOR["Advertencia"]],
                    [0.6666, LEVEL_COLOR["Advertencia"]], [0.6667, LEVEL_COLOR["Crítico"]], [1, LEVEL_COLOR["Crítico"]]],
        hovertemplate="Hora %{x:.2f} h · %{y}<br>Estado: %{text}<extra></extra>"))

    def first_warning(alerts):
        if alerts is None or alerts.empty:
            return None
        rows = alerts[~alerts["Nivel"].map(is_intervention)]
        return float(rows.iloc[0]["Hora"]) if len(rows) else None

    for label, alerts in (("Reactiva", alerts_re), ("Predictiva", alerts_pr)):
        h = first_warning(alerts)
        if h is not None:
            fig.add_trace(go.Scatter(x=[h], y=[label], mode="markers", name="Primera alerta", showlegend=False,
                                     marker=dict(symbol="star", size=12, color="white", line=dict(color="black", width=1)),
                                     hovertemplate=f"Primera alerta: {h:.2f} h<extra></extra>"))
    if idx_pr is not None:
        h = float(df_pr["Hora simulada"].iloc[idx_pr])
        fig.add_trace(go.Scatter(x=[h], y=["Predictiva"], mode="markers", showlegend=False,
                                 marker=dict(symbol="diamond", size=12, color="#0b1a2a", line=dict(color="white", width=2)),
                                 hovertemplate=f"Intervención propuesta: {h:.2f} h<extra></extra>"))
    fig.add_vline(x=hour, line_color="#e2e8f0", line_width=2)
    fig.update_layout(height=190, margin=dict(l=10, r=10, t=10, b=30), paper_bgcolor="#0b1a2a", plot_bgcolor="#081522",
                      font_color="#b8c7d9", xaxis_title="Hora simulada", yaxis=dict(autorange="reversed"))
    return fig

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
    fig.update_layout(height=440, **THEME,
                      margin=dict(l=20, r=20, t=30, b=20),
                      xaxis_title="Incremento del MTBF (%)", yaxis_title="Reducción del MTTR (%)",
                      legend=dict(orientation="h", y=1.1))
    return fig


# ===========================================================================
# REPLICA3D
# ===========================================================================

_LIGHT = dict(ambient=0.62, diffuse=0.85, specular=0.18, roughness=0.6)

# Paleta de los equipos
RED, HAZ_Y, HAZ_K = "#b3262c", "#e9b800", "#1b1d21"
STEEL, DARK, RUST, SLAT = "#9aa5b1", "#3a414b", "#6d4c3d", "#7b5a47"
CREAM, PANEL, GREEN_FRAME = "#e6e1d1", "#dfe5e8", "#3f8f5a"
PLASMA_BLUE, ORANGE, PROPOSED = "#1f6fbf", "#e8632b", "#14b8a6"

_BOX_I = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2]
_BOX_J = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3]
_BOX_K = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6]


def boxes_mesh(boxes, color, name, opacity=1.0):
    """Varias cajas (x0, y0, z0, dx, dy, dz) combinadas en una sola malla."""
    xs, ys, zs, ii, jj, kk = [], [], [], [], [], []
    for n, (x0, y0, z0, dx, dy, dz) in enumerate(boxes):
        xs += list(x0 + dx * np.array([0, 0, 1, 1, 0, 0, 1, 1]))
        ys += list(y0 + dy * np.array([0, 1, 1, 0, 0, 1, 1, 0]))
        zs += list(z0 + dz * np.array([0, 0, 0, 0, 1, 1, 1, 1]))
        ii += [v + 8 * n for v in _BOX_I]
        jj += [v + 8 * n for v in _BOX_J]
        kk += [v + 8 * n for v in _BOX_K]
    return go.Mesh3d(x=xs, y=ys, z=zs, i=ii, j=jj, k=kk, color=color, opacity=opacity, name=name,
                     hoverinfo="name", flatshading=True, lighting=_LIGHT, showlegend=False)


def box_mesh(x0, y0, z0, dx, dy, dz, color, name, opacity=1.0):
    return boxes_mesh([(x0, y0, z0, dx, dy, dz)], color, name, opacity)


def cyl_mesh(axis, start, length, radius, c1, c2, color, name, n=36, opacity=1.0):
    """Cilindro con tapas. axis='x': centro (y=c1, z=c2); 'y': (x=c1, z=c2); 'z': (x=c1, y=c2)."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    u, v = c1 + radius * np.cos(t), c2 + radius * np.sin(t)
    a = np.r_[np.full(n, start), np.full(n, start + length), [start, start + length]]
    p = np.r_[u, u, [c1, c1]]
    q = np.r_[v, v, [c2, c2]]
    if axis == "x":
        x, y, z = a, p, q
    elif axis == "y":
        x, y, z = p, a, q
    else:
        x, y, z = p, q, a
    c0, c1i = 2 * n, 2 * n + 1
    i, j, k = [], [], []
    for s in range(n):
        b = (s + 1) % n
        i += [s, b, c0, c1i]
        j += [b, b + n, b, s + n]
        k += [s + n, s + n, s, b + n]
    return go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color=color, opacity=opacity, name=name,
                     hoverinfo="name", flatshading=True, lighting=_LIGHT, showlegend=False)


def hazard_boxes(x0, y0, z0, dx, dy, dz, n=8):
    """Franjas amarillas y negras alternadas (base del pórtico)."""
    step = dx / n
    yellow = [(x0 + s * step, y0, z0, step, dy, dz) for s in range(0, n, 2)]
    black = [(x0 + s * step, y0, z0, step, dy, dz) for s in range(1, n, 2)]
    return yellow, black



def tint(box, status_color, name):
    """Capa translúcida del color de estado sobre una pieza monitoreada."""
    x0, y0, z0, dx, dy, dz = box
    return box_mesh(x0 - 12, y0 - 12, z0 - 12, dx + 24, dy + 24, dz + 24, status_color, name, opacity=0.26)


def _cnc_scene(status_color):
    L, W, top = 6000, 3000, 800      # recorrido (X), luz del pórtico (Y) y altura de la mesa
    yc = 1250                        # posición ilustrativa del carro sobre la viga
    parts = []
    # Mesa de parrilla con láminas
    parts.append(box_mesh(0, 0, 400, L, W, 300, RUST, "Mesa de corte (cuerpo)"))
    parts.append(boxes_mesh([(0, -20, 400, L, 20, 420), (0, W, 400, L, 20, 420),
                             (-20, 0, 400, 20, W, 420), (L, 0, 400, 20, W, 420)], "#5a3f33", "Paredes de la mesa"))
    parts.append(boxes_mesh([(float(x), 0, 700, 10, W, 100) for x in np.arange(60, L - 60, 100)], SLAT, "Láminas de la mesa"))
    parts.append(boxes_mesh([(x, y, 0, 120, 120, 400) for x in (100, 2900, 5700) for y in (60, W - 180)], DARK, "Patas de la mesa"))
    parts.append(box_mesh(1800, 700, top, 2600, 1400, 25, "#2c333b", "Chapa en proceso (referencial)"))
    # Rieles de desplazamiento del pórtico (a ambos lados de la mesa)
    for y0, name in ((-520, "izquierdo"), (W + 320, "derecho")):
        parts.append(box_mesh(-600, y0, 0, 7200, 200, 300, DARK, f"Soporte del riel {name}"))
        parts.append(box_mesh(-600, y0 + 50, 300, 7200, 100, 50, STEEL, f"Riel {name}"))
    # Carro izquierdo (lado de la fuente de plasma) con base de franjas de seguridad
    yel, blk = hazard_boxes(300, -600, 350, 800, 360, 550)
    parts += [boxes_mesh(yel, HAZ_Y, "Base del carro izquierdo (franjas)"), boxes_mesh(blk, HAZ_K, "Base del carro izquierdo (franjas)")]
    left_house = (400, -560, 900, 600, 300, 600)
    parts.append(box_mesh(*left_house, RED, "Carro izquierdo del pórtico"))
    # Carro derecho (lado del pupitre) con armario eléctrico
    yel, blk = hazard_boxes(300, W + 240, 350, 800, 360, 550)
    parts += [boxes_mesh(yel, HAZ_Y, "Base del carro derecho (franjas)"), boxes_mesh(blk, HAZ_K, "Base del carro derecho (franjas)")]
    right_house = (400, W + 260, 900, 600, 300, 800)
    parts.append(box_mesh(*right_house, RED, "Carro derecho del pórtico"))
    parts.append(box_mesh(400, W + 270, 1700, 600, 300, 850, "#d8dcdc", "Armario eléctrico"))
    # Viga, cremallera y cadena portacables
    parts.append(box_mesh(500, -330, 1380, 450, W + 660, 320, RED, "Viga del pórtico"))
    parts.append(box_mesh(560, -250, 1700, 330, W + 500, 30, STEEL, "Cremallera y guía superior"))
    parts.append(box_mesh(640, 1600, 1730, 160, W - 1700, 70, "#20242a", "Cadena portacables"))
    parts.append(box_mesh(640, 1600, 1800, 160, W - 1700, 12, HAZ_Y, "Cadena portacables (eslabones)"))
    parts.append(box_mesh(520, 100, 1730, 420, 800, 840, PLASMA_BLUE, "Fuente de plasma LGK-200 IGBT"))
    parts.append(box_mesh(540, 1000, 1730, 360, 450, 300, ORANGE, "Equipo auxiliar sobre la viga (caja naranja)"))
    # Carro de antorchas con dos elevadores (plasma y oxicorte)
    parts.append(box_mesh(950, yc - 300, 1400, 330, 600, 360, DARK, "Carro de antorchas"))
    parts.append(box_mesh(1280, yc - 230, 820, 160, 150, 780, "#2b3038", "Elevador de la antorcha de plasma"))
    parts.append(box_mesh(1280, yc + 80, 820, 160, 150, 780, "#2b3038", "Elevador de la antorcha de oxicorte"))
    parts.append(cyl_mesh("z", 835, 560, 28, 1360, yc - 155, "#c9722f", "Antorcha de plasma"))
    parts.append(cyl_mesh("z", 835, 560, 26, 1360, yc + 155, "#8a8f98", "Antorcha de oxicorte"))
    # Pupitre del controlador F2300BT
    parts.append(box_mesh(950, W + 60, 1250, 100, 200, 300, RED, "Soporte del pupitre"))
    parts.append(box_mesh(1000, W - 250, 1250, 350, 400, 500, "#cfcfc8", "Pupitre del controlador"))
    parts.append(box_mesh(1350, W - 200, 1480, 12, 300, 200, "#0b1220", "Pantalla F2300BT"))
    parts.append(box_mesh(1350, W - 210, 1270, 12, 320, 190, "#3f4650", "Teclado F2300BT"))
    parts.append(box_mesh(1000, W - 250, 1100, 380, 400, 150, "#b9b9b2", "Panel de mandos"))
    parts.append(cyl_mesh("z", 1250, 40, 30, 1250, W + 60, "#d12a2a", "Paro de emergencia"))
    for kx, ky in ((1080, W - 150), (1080, W - 30), (1180, W - 150), (1180, W - 30)):
        parts.append(cyl_mesh("z", 1250, 28, 22, kx, ky, "#2a2d31", "Selector del panel"))
    # Elementos propuestos (aún no instalados): gabinete de adquisición
    parts.append(box_mesh(1250, W + 700, 0, 450, 300, 900, PROPOSED, "Gabinete DAQ y gateway (propuesto)", opacity=0.55))
    # Capas de estado sobre las piezas monitoreadas
    parts.append(tint(left_house, status_color, "Estado: accionamiento izquierdo"))
    parts.append(tint(right_house, status_color, "Estado: accionamiento derecho"))
    sensors = [
        ("Vibración (mm/s RMS)", "Vibración eje X (base)", (1030, W + 410, 1150)),
        ("Vibración (mm/s RMS)", "Vibración eje Y (contingencia)", (1290, yc, 1600), True),
        ("Temperatura (°C)", "Temp. servo X", (1030, -410, 1150)),
        ("Temperatura (°C)", "Temp. servo Y", (1010, W + 300, 1600)),
        ("Corriente (A)", "Corriente", (1020, W + 420, 2150)),
        ("Error de posición (mm)", "Error de posición (inspección TPM)", (1360, yc - 155, 850)),
    ]
    beacon = (700, W + 420, 2800)
    return parts, sensors, beacon, dict(x=1.7, y=-1.35, z=0.85)


def _lathe_scene(status_color):
    parts = []
    # Bancada con franja roja y base
    parts.append(box_mesh(0, 150, 450, 3600, 600, 500, "#1f2328", "Bancada"))
    parts.append(box_mesh(0, 144, 640, 3600, 8, 60, RED, "Franja roja de la bancada"))
    parts.append(box_mesh(0, 144, 705, 3600, 8, 14, STEEL, "Franja plateada"))
    parts.append(boxes_mesh([(80, 160, 0, 700, 580, 450), (2850, 160, 0, 700, 580, 450)], "#23272d", "Base de la bancada"))
    parts.append(box_mesh(250, 138, 500, 420, 8, 120, "#ececec", "Placa TOPTECH"))
    # Cabezal con paneles
    head = (0, 100, 950, 1100, 700, 550)
    parts.append(box_mesh(*head, CREAM, "Cabezal"))
    parts.append(box_mesh(70, 94, 1180, 780, 6, 290, GREEN_FRAME, "Marco del panel de palancas"))
    parts.append(box_mesh(80, 86, 1190, 760, 8, 270, PANEL, "Panel de palancas del cabezal"))
    parts.append(box_mesh(70, 94, 965, 660, 6, 200, GREEN_FRAME, "Marco del panel de avances"))
    parts.append(box_mesh(80, 86, 975, 640, 8, 180, PANEL, "Panel de avances"))
    parts.append(cyl_mesh("y", 76, 12, 108, 880, 1330, "#c9a227", "Disco selector de velocidades"))
    parts.append(cyl_mesh("y", 68, 10, 86, 880, 1330, "#f1f1f1", "Disco selector de velocidades"))
    parts.append(cyl_mesh("y", 50, 22, 34, 880, 1330, STEEL, "Palanca de velocidades"))
    parts.append(cyl_mesh("y", 70, 24, 30, 1040, 1210, "#d12a2a", "Paro de emergencia"))
    parts.append(cyl_mesh("y", 70, 16, 18, 1040, 1120, "#22c55e", "Piloto verde"))
    # Plato de garras y pieza
    parts.append(cyl_mesh("x", 1100, 150, 235, 450, 1265, "#3b4047", "Plato de garras"))
    parts.append(boxes_mesh([(1250, 210, 1225, 110, 90, 80), (1250, 600, 1225, 110, 90, 80),
                             (1250, 410, 1415, 110, 80, 90), (1250, 410, 1035, 110, 80, 90)], "#5a6068", "Garras del plato"))
    parts.append(cyl_mesh("x", 1360, 150, 80, 450, 1265, "#c6ccd3", "Pieza de trabajo (referencial)"))
    # Respaldo, tornillo patrón y barra de avance
    parts.append(box_mesh(1100, 760, 950, 2400, 60, 420, CREAM, "Respaldo de bancada"))
    parts.append(cyl_mesh("x", 1100, 2400, 22, 120, 810, "#aab4bf", "Tornillo patrón"))
    parts.append(cyl_mesh("x", 1100, 2400, 16, 120, 880, "#aab4bf", "Barra de avance"))
    # Carro longitudinal, delantal con volantes, carro transversal y portaherramientas de 4 posiciones
    parts.append(box_mesh(1500, 170, 950, 640, 560, 110, "#2c3138", "Carro longitudinal"))
    parts.append(box_mesh(1500, 90, 760, 640, 90, 330, "#e6e1d1", "Delantal del carro"))
    parts.append(cyl_mesh("y", 20, 70, 140, 1700, 880, "#b8c0c8", "Volante del carro"))
    parts.append(cyl_mesh("y", 20, 70, 100, 1950, 900, "#b8c0c8", "Volante transversal"))
    parts.append(box_mesh(1560, 230, 1060, 480, 500, 70, "#2f353c", "Carro transversal"))
    parts.append(box_mesh(1620, 330, 1130, 300, 260, 60, "#cfd4d9", "Carro auxiliar"))
    parts.append(box_mesh(1650, 370, 1190, 220, 200, 120, "#3d434a", "Portaherramientas de 4 posiciones"))
    for bx, by in ((1670, 390), (1850, 390), (1670, 550), (1850, 550)):
        parts.append(cyl_mesh("z", 1310, 30, 14, bx, by, "#aeb6bf", "Tornillo de sujeción"))
    parts.append(box_mesh(1510, 440, 1255, 150, 22, 22, "#ff8a3d", "Herramienta de corte"))
    # Contrapunto
    parts.append(box_mesh(2750, 200, 950, 520, 500, 120, "#2f353c", "Base del contrapunto"))
    parts.append(box_mesh(2820, 260, 1070, 380, 380, 380, CREAM, "Contrapunto"))
    parts.append(cyl_mesh("x", 2680, 140, 42, 450, 1265, "#aeb6bf", "Caña del contrapunto"))
    parts.append(cyl_mesh("x", 2630, 50, 18, 450, 1265, "#c9d0d6", "Punto giratorio"))
    parts.append(cyl_mesh("x", 3200, 30, 120, 450, 1265, "#b8c0c8", "Volante del contrapunto"))
    # Elemento propuesto (aún no instalado): gabinete de adquisición
    parts.append(box_mesh(-700, 300, 0, 450, 300, 900, PROPOSED, "Gabinete DAQ y gateway (propuesto)", opacity=0.55))
    parts.append(tint(head, status_color, "Estado: cabezal y accionamiento"))
    sensors = [
        ("Vibración (mm/s RMS)", "Vibración cabezal (base)", (350, 450, 1530)),
        ("Vibración (mm/s RMS)", "Vibración carro (contingencia)", (1760, 470, 1340), True),
        ("Temperatura (°C)", "Temp. cabezal y caja", (-30, 450, 1150)),
        ("Temperatura (°C)", "Temp. motor", (-30, 450, 600)),
        ("Corriente (A)", "Corriente", (-475, 450, 960)),
        ("RPM", "RPM", (1120, 450, 1540)),
    ]
    beacon = (550, 450, 1800)
    return parts, sensors, beacon, dict(x=0.35, y=-1.95, z=0.65)


def replica_figure(asset_name, readings):
    """Modelo 3D esquemático del equipo con sus puntos de medición.
    Los colores reales se conservan; el estado de las lecturas se muestra en las capas
    translúcidas sobre las piezas monitoreadas, en los marcadores de sensores y en el indicador superior."""
    asset = ASSETS[asset_name]
    levels = {v: reading_level(readings[v][0], asset["variables"][v]) for v in asset["variables"]}
    equipment_level = worst_level(list(levels.values()))
    status_color = LEVEL_COLOR[equipment_level]
    parts, sensor_points, beacon, eye = _cnc_scene(status_color) if asset_name == "CNC plasma" else _lathe_scene(status_color)

    sx, sy, sz, text, hover, colors, symbols = [], [], [], [], [], [], []
    for point in sensor_points:
        variable, label, (x, y, z) = point[:3]
        optional = len(point) > 3 and point[3]          # punto de contingencia (se instala solo si se cumple el criterio)
        value, _ = readings[variable]
        level = levels[variable]
        sx.append(x); sy.append(y); sz.append(z)
        text.append(label)
        hover.append(f"{variable}: {value_text(variable, value)} · {level}" + (" · punto de contingencia" if optional else ""))
        colors.append(LEVEL_COLOR[level])
        symbols.append("diamond-open" if optional else "diamond")
    sensors = go.Scatter3d(
        x=sx, y=sy, z=sz, mode="markers+text", text=text, textposition="top center",
        textfont=dict(color="#e2e8f0", size=11), hovertext=hover, hoverinfo="text", showlegend=False,
        marker=dict(size=8, color=colors, symbol=symbols, line=dict(width=2, color="#0f172a")),
    )
    indicator = go.Scatter3d(
        x=[beacon[0]], y=[beacon[1]], z=[beacon[2]], mode="markers+text", text=[f"Estado: {equipment_level}"],
        textposition="top center", textfont=dict(color="#e2e8f0", size=12), hoverinfo="skip", showlegend=False,
        marker=dict(size=14, color=status_color, symbol="circle", line=dict(width=2, color="#0f172a")),
    )
    fig = go.Figure(parts + [sensors, indicator])
    fig.update_layout(
        height=600, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor=PAPER_BG, font_color=FONT_COLOR,
        scene=dict(aspectmode="data", bgcolor=PLOT_BG, camera=dict(eye=eye),
                   xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
    )
    return fig, levels


# ===========================================================================
# CACHING
# ===========================================================================

@st.cache_data(show_spinner=False, max_entries=64)
def simulate_telemetry(sim: SimConfig, policy: str):
    return simulate_frames(sim, policy)


@st.cache_data(show_spinner=False, max_entries=16)
def experiment_results(sim: SimConfig) -> pd.DataFrame:
    return experiment_matrix(sim, simulate=simulate_telemetry)


@st.cache_data(show_spinner=False, max_entries=32)
def monte_carlo(asset_name, mtbf_gain, mttr_reduction, simulations, seed):
    return monte_carlo_sample(asset_name, mtbf_gain, mttr_reduction, simulations, seed)


@st.cache_data(show_spinner=False, max_entries=8)
def seed_robustness(asset_name, mtbf_gain, mttr_reduction, simulations, n_seeds=100):
    return seed_means(asset_name, mtbf_gain, mttr_reduction, simulations, n_seeds)


@st.cache_data(show_spinner=False, max_entries=16)
def _critical_hours(ref: SimConfig, n: int) -> np.ndarray:
    return critical_hours(ref, n)


@st.cache_data(show_spinner=False, max_entries=32)
def _intervention_hours(sim: SimConfig, critical: np.ndarray) -> np.ndarray:
    return intervention_hours(sim, critical)


@st.cache_data(show_spinner=False, max_entries=32)
def _false_alarm_rate(sim: SimConfig, n: int) -> float:
    return false_alarm_rate(sim, n)


def detection_performance(sim: SimConfig, min_lead_h: float, n: int = DEFAULT_TRAJECTORIES) -> dict:
    """Desempeño de la detección. Cambiar `min_lead_h` no vuelve a simular; cambiar el umbral reutiliza la referencia."""
    critical = _critical_hours(reference_config(sim), n)
    interventions = _intervention_hours(detection_config(sim), critical)
    false_alarms = _false_alarm_rate(replace(sim, scenario="Operación normal", seed=0, evaluate_normal=True), n)
    return summarize_detection(critical, interventions, false_alarms, min_lead_h)


def threshold_tradeoff(sim: SimConfig, min_lead_h: float, n: int = 60) -> pd.DataFrame:
    """Compromiso del umbral de intervención con el método por lecturas: anticipación frente a falsas alarmas."""
    rows = []
    for pct in range(45, 86, 5):
        trial = replace(sim, intervention_threshold=pct / 100, detection=DETECCION_LECTURAS)
        r = detection_performance(trial, min_lead_h, n)
        rows.append({"Umbral (%)": pct, "Fallas anticipadas (%)": r["p_anticipada"] * 100,
                     "Falsas alarmas (%)": r["falsas_alarmas"] * 100, "Aviso mediano (h)": r["aviso_mediano_h"]})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False, max_entries=8)
def analyze_csv(csv_bytes: bytes, asset_name: str, intervention_threshold: float, thresholds_id: str):
    """Interpreta y analiza un CSV de lecturas reales (el resultado se reutiliza mientras no cambie nada)."""
    return analyze_readings(parse_csv(csv_bytes, asset_name), asset_name, intervention_threshold)


# ===========================================================================
# SIDEBAR
# ===========================================================================

@dataclass(frozen=True)
class Controls:
    asset_name: str
    scenario: str
    policy: str
    duration: int
    seed: int
    mtbf_gain: float
    mttr_reduction: float
    simulations: int
    noise_factor: float
    missing_rate_pct: int
    intervention_pct: int
    position_capture: str
    inspection_h: int
    load_pct: int
    failure_mode: str
    compensate_load: bool
    detection: str
    data_source: str = "Simulación"
    csv_bytes: Optional[bytes] = None

    @property
    def missing_rate(self) -> float:
        return self.missing_rate_pct / 100

    @property
    def intervention_threshold(self) -> float:
        return self.intervention_pct / 100

    @property
    def inspection_eff(self) -> float:
        """Intervalo efectivo de inspección (0 = captura continua)."""
        return float(self.inspection_h) if self.position_capture == CAPTURA_INSPECCION else 0.0


def init_state() -> None:
    for key, value in DEFAULTS.items():
        st.session_state.setdefault(key, value)


def reset_defaults() -> None:
    for key, value in DEFAULTS.items():
        st.session_state[key] = value


def render_sidebar() -> Controls:
    init_state()
    with st.sidebar:
        st.markdown("## ⚙️ Configuración")
        data_source = st.radio("Fuente de datos", DATA_SOURCES, horizontal=True, key="data_source",
                               help="Simulación: señales sintéticas para evaluar el gemelo. Datos reales: lecturas de sensores cargadas desde un CSV.")
        asset_name = st.selectbox("Activo digital", list(ASSETS.keys()), key="asset_name")
        csv_bytes = None
        if data_source == DATOS_REALES:
            upload = st.file_uploader("Lecturas de sensores (CSV)", type=["csv"], key="csv_file")
            csv_bytes = upload.getvalue() if upload is not None else None
            st.download_button("⬇ Plantilla de CSV", data=template_csv(asset_name), file_name=f"plantilla_{ASSETS[asset_name]['code'].lower()}.csv",
                               mime="text/csv", help="Ejemplo con el formato esperado: una columna de tiempo y una por variable.")
            st.caption("Con datos reales solo se aplican el equipo y el umbral de intervención. Los controles de simulación no tienen efecto.")
        scenario = st.selectbox("Escenario", SCENARIOS, key="scenario")
        policy = st.radio("Política de mantenimiento", ["Reactivo", "Predictivo"], horizontal=True, key="policy")
        duration = st.slider("Duración simulada (horas)", 8, 72, step=4, key="duration")
        seed = st.number_input("Semilla del experimento", min_value=1, max_value=9999, key="seed")
        st.divider()
        st.markdown("#### Supuestos de mejora (no medidos)")
        mtbf_gain = st.slider("Incremento proyectado del MTBF (%)", 5.0, 60.0, step=0.1, format="%.1f", key="mtbf_gain")
        mttr_reduction = st.slider("Reducción proyectada del MTTR (%)", 5.0, 60.0, step=0.1, format="%.1f", key="mttr_reduction")
        simulations = st.select_slider("Corridas Monte Carlo", options=[100, 250, 500, 1000, 5000, 10000], key="simulations")
        st.divider()
        st.markdown("#### Pruebas de robustez")
        noise_factor = st.slider("Ruido del sensor", 0.5, 3.0, step=0.1, key="noise_factor")
        missing_rate_pct = st.slider("Pérdida simulada de datos (%)", 0, 20, step=1, key="missing_rate_pct")
        intervention_pct = st.slider("Umbral de intervención (% de degradación)", 45, 85, step=1, key="intervention_pct")
        position_capture = st.selectbox(
            "Captura del error de posición (CNC)", [CAPTURA_CONTINUA, CAPTURA_INSPECCION], key="position_capture",
            help="El controlador F2300BT no exporta el error de posición: en planta se mide en la inspección del mantenimiento autónomo. "
                 "«Continua» es una simplificación del modelo.")
        inspection_h = st.slider("Intervalo entre inspecciones TPM (h)", 1, 24, step=1, key="inspection_h",
                                 disabled=position_capture == CAPTURA_CONTINUA)
        load_pct = st.slider("Variación de la carga operativa (%)", 0, 100, step=5, key="load_pct",
                             help="Simula cambios de carga que mueven la vibración, la corriente y la temperatura sin que haya degradación. "
                                  "Con 100 %, a plena carga la corriente llega al límite crítico.")
        failure_mode = st.selectbox("Modo de falla simulado", list(MODOS_FALLA.keys()), key="failure_mode",
                                    help="Qué variables muestran la degradación. En el CNC, el modo «solo error de posición» representa las holguras "
                                         "y descalibraciones del historial, que solo se ven en la inspección. Con el método de estado interno "
                                         "el umbral se evalúa sobre el estado interno y este modo no cambia la intervención.")
        compensate_load = st.checkbox("Compensar el régimen de carga (perfil normal por régimen)", key="compensate_load",
                                      help="Solo aplica al método «estimado desde lecturas»: el gemelo descuenta la carga conocida antes de estimar la condición.")
        detection = st.selectbox(
            "Método de detección", [DETECCION_ESTADO, DETECCION_LECTURAS], key="detection",
            help="'Estado interno': el umbral se evalúa sobre el estado interno del simulador. "
                 "'Estimado desde lecturas': el gemelo estima la condición solo con las lecturas de los sensores "
                 "y exige 4 lecturas seguidas por encima del umbral.",
        )
        st.button("↺ Restaurar valores por defecto", on_click=reset_defaults, width="stretch")
        st.caption("Los valores por defecto reproducen la configuración principal. "
                   "Los supuestos pueden modificarse para el análisis de sensibilidad.")
        with st.expander("¿Qué afecta a qué?"):
            st.markdown(
                "**Cambian la franja de estado, la telemetría, las alertas y la réplica:** activo, escenario, duración, semilla, "
                "ruido, pérdida de datos, umbral de intervención, método de detección, captura del error de posición, variación de carga y compensación de régimen.\n\n"
                "**Cambian solo los indicadores (Monte Carlo y sensibilidad):** incremento del MTBF, reducción del MTTR y corridas.\n\n"
                "**Los vincula** el panel «Sustento del supuesto» de la pestaña Sensibilidad: calcula cuántas fallas debe evitar "
                "el gemelo para sostener el incremento del MTBF y lo compara con lo que logra la detección simulada."
            )
    return Controls(
        asset_name=asset_name, scenario=scenario, policy=policy, duration=int(duration), seed=int(seed),
        mtbf_gain=float(mtbf_gain), mttr_reduction=float(mttr_reduction), simulations=int(simulations),
        noise_factor=float(noise_factor), missing_rate_pct=int(missing_rate_pct), intervention_pct=int(intervention_pct),
        position_capture=position_capture, inspection_h=int(inspection_h), load_pct=int(load_pct),
        failure_mode=failure_mode, compensate_load=bool(compensate_load), detection=detection,
        data_source=data_source, csv_bytes=csv_bytes,
    )


# ===========================================================================
# VISTA · TELEMETRY
# ===========================================================================

def render_telemetry(c):
    (asset, asset_name, df, hour_now, inspection_eff, readings) = (c.asset, c.asset_name, c.df, c.hour_now, c.inspection_eff, c.readings)
    st.markdown('<div class="section-title">Comportamiento virtual del activo</div>', unsafe_allow_html=True)
    st.caption("Línea amarilla: umbral de advertencia · Línea roja: umbral crítico · Línea gris vertical: instante elegido")
    st.plotly_chart(telemetry_figure(df, asset_name, hour_now, inspection_eff > 0), width="stretch", config=PLOTLY_CONFIG)
    columns = st.columns(4)
    for column, variable in zip(columns, asset["variables"]):
        value, age = readings[variable]
        with column:
            reading_card(variable, value, asset["variables"][variable], age, inspection_eff > 0)
    st.download_button(
        "⬇ Descargar telemetría CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"telemetria_{asset['code'].lower()}.csv",
        mime="text/csv",
    )


# ===========================================================================
# VISTA · REPLICA
# ===========================================================================

def render_replica(c):
    (asset, asset_name, df, health_now, hour_now, inspection_eff, moment, readings, status_now) = (c.asset, c.asset_name, c.df, c.health_now, c.hour_now, c.inspection_eff, c.moment, c.readings, c.status_now)
    st.markdown('<div class="section-title">Réplica virtual del equipo</div>', unsafe_allow_html=True)
    st.caption("Los marcadores indican los "
               "puntos de medición de la propuesta de instrumentación; su color, y el de las capas translúcidas sobre las piezas "
               "monitoreadas, refleja el estado de la lectura simulada en el instante elegido. El gabinete verde azulado es un "
               "elemento propuesto, aún no instalado. Los marcadores abiertos son los acelerómetros de contingencia: se instalan solo si se cumple el criterio de activación.")
    left, right = st.columns([2.1, 1])
    with left:
        figure, levels = replica_figure(asset_name, readings)
        st.plotly_chart(figure, width="stretch", config=PLOTLY_CONFIG)
        legend = "".join(
            f'<span class="legend-dot" style="background:{LEVEL_COLOR[name]}"></span>{name}'
            for name in ["Normal", "Advertencia", "Crítico", "Sin dato"]
        )
        st.markdown(f'<div class="small-muted">Estado del punto de medición:{legend}'
                    f'<span class="legend-dot" style="background:#14b8a6"></span>Elemento propuesto</div>', unsafe_allow_html=True)
    with right:
        estimate_now = float(df.iloc[moment]["Índice de condición estimado (%)"])
        st.markdown('<div class="section-title">Ficha digital del equipo</div>', unsafe_allow_html=True)
        ficha = pd.DataFrame({
            "Atributo": ["Identificación", "Tipo de elemento", "Área", "Estado actual", "Hora simulada",
                         "Índice de salud (simulador)", "Condición estimada desde lecturas", "Perfil normal vigente",
                         "Fuente de datos", "Actualización", "Rol del gemelo"],
            "Valor": [asset["code"], "Equipo (elemento de manufactura observable)", asset["area"], status_now,
                      f"{hour_now:.2f} h", f"{health_now:.1f} %", f"{estimate_now:.1f} %", str(df.iloc[moment]["Perfil normal"]),
                      "Señal sintética (simulación)", "Cada 5 min (casi en tiempo real)",
                      "Consultivo: propone la OT; decide el jefe de Mantenimiento"],
        })
        st.dataframe(ficha, hide_index=True, width="stretch")
    st.markdown('<div class="section-title">Equipo real que representa la réplica</div>', unsafe_allow_html=True)
    specs = pd.DataFrame(EQUIPMENT_SPECS[asset_name], columns=["Característica", "Detalle", "Fuente"])
    st.dataframe(specs, hide_index=True, width="stretch")
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
            "Origen en esta simulación": (f"Señal sintética tomada cada {inspection_eff:g} h (inspección TPM)" if cfg.get("manual") and inspection_eff > 0 else "Señal sintética cada 5 min, identificada como simulada"),
            "Origen en la implementación": origin,
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Los umbrales son demostrativos y se calibrarán con la línea base real y las especificaciones del fabricante. "
               "Las cantidades de instrumentación están en el plan de implementación.")


# ===========================================================================
# VISTA · INDICATORS
# ===========================================================================

def render_indicators(c):
    (alerts_pr, asset_name, det_availability, det_mtbf, df_pr, df_re, idx_pr, inspection_eff, mc, scenario, seed, simulations) = (c.alerts_pr, c.asset_name, c.det_availability, c.det_mtbf, c.df_pr, c.df_re, c.idx_pr, c.inspection_eff, monte_carlo(c.asset_name, c.mtbf_gain, c.mttr_reduction, c.simulations, c.seed), c.scenario, c.seed, c.simulations)
    if c.is_real:
        st.info("Con datos reales, los criterios de detección usan tus lecturas. La simulación Monte Carlo sigue basándose en los supuestos de mejora de la barra lateral, no en mediciones.")
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
    fig_box.update_layout(height=430, **THEME, margin=dict(l=20, r=20, t=55, b=20))
    fig_box.update_yaxes(gridcolor=GRID_COLOR)
    st.plotly_chart(fig_box, width="stretch", config=PLOTLY_CONFIG)
    st.caption(f"Línea verde: meta de disponibilidad ({META_DISPONIBILIDAD} %).")

    st.markdown('<div class="section-title">Criterios de aceptación (en simulación)</div>', unsafe_allow_html=True)
    predictive = mc[mc["Política"] == "Predictiva"]
    mc_availability = predictive["Disponibilidad (%)"].mean()
    mc_mtbf = predictive["MTBF (h)"].mean()
    first_alert_row = alerts_pr[~alerts_pr["Nivel"].map(is_intervention)] if not alerts_pr.empty else alerts_pr
    first_alert_hour = float(first_alert_row.iloc[0]["Hora"]) if len(first_alert_row) else np.nan
    critical_reactive = df_re.index[df_re["Estado"] == "Crítico"].tolist()
    critical_hour_reactive = float(df_re.loc[critical_reactive[0], "Hora simulada"]) if critical_reactive else np.nan
    predictive_critical = bool((df_pr["Estado"] == "Crítico").any())
    intervention_hour = float(df_pr.loc[idx_pr, "Hora simulada"]) if idx_pr is not None else np.nan

    criteria = [
        ("Calidad de datos", f"≥ {META_CALIDAD_DATOS:.0f} %", f"{data_quality(df_pr, asset_name, inspection_eff > 0):.2f} %", verdict(data_quality(df_pr, asset_name, inspection_eff > 0) >= META_CALIDAD_DATOS)),
        ("Disponibilidad determinística (supuestos actuales)", f"≥ {META_DISPONIBILIDAD} %", f"{det_availability:.2f} %", verdict(det_availability >= META_DISPONIBILIDAD)),
        ("Disponibilidad media Monte Carlo (predictiva)", f"≥ {META_DISPONIBILIDAD} %", f"{mc_availability:.2f} %", verdict(mc_availability >= META_DISPONIBILIDAD)),
        ("MTBF supuesto del escenario (parámetro de entrada)", f"> {META_MTBF} h", f"{det_mtbf:.2f} h", verdict(det_mtbf > META_MTBF)),
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


# ===========================================================================
# VISTA · SENSITIVITY
# ===========================================================================

def render_sensitivity(c):
    (asset_name, base_mtbf, base_mttr, det_availability, det_mtbf, detection, mtbf_gain, mttr_reduction, scenario, seed, sim, simulations) = (c.asset_name, c.base_mtbf, c.base_mttr, c.det_availability, c.det_mtbf, c.detection, c.mtbf_gain, c.mttr_reduction, c.scenario, c.seed, c.sim, c.simulations)
    st.markdown('<div class="section-title">Qué supuestos hacen falta para cumplir las metas</div>', unsafe_allow_html=True)
    st.caption("Disponibilidad determinística = MTBF / (MTBF + MTTR). La línea roja marca la meta de disponibilidad; el punto blanco, el supuesto actual.")
    st.plotly_chart(sensitivity_figure(asset_name, mtbf_gain, mttr_reduction), width="stretch", config=PLOTLY_CONFIG)

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
    sims_rob = min(int(simulations), 1000)
    robust = seed_robustness(asset_name, mtbf_gain, mttr_reduction, sims_rob)
    share_mtbf = (robust["MTBF (h)"] > META_MTBF).mean() * 100
    share_av = (robust["Disponibilidad (%)"] >= META_DISPONIBILIDAD).mean() * 100
    current = monte_carlo(asset_name, mtbf_gain, mttr_reduction, sims_rob, int(seed))
    current_mtbf = current[current["Política"] == "Predictiva"]["MTBF (h)"].mean()
    percentile = (robust["MTBF (h)"] < current_mtbf).mean() * 100
    c1, c2, c3 = st.columns(3)
    c1.metric(f"MTBF medio entre 100 semillas ({sims_rob:,} corridas c/u)", f"{robust['MTBF (h)'].mean():.2f} h", f"Valor esperado: {det_mtbf:.2f} h", delta_color="off")
    c2.metric(f"Semillas con MTBF > {META_MTBF} h", f"{share_mtbf:.0f} %", f"Semilla {int(seed)}: {current_mtbf:.2f} h (percentil {percentile:.0f})", delta_color="off")
    c3.metric(f"Semillas con disponibilidad ≥ {META_DISPONIBILIDAD} %", f"{share_av:.0f} %", f"Promedio: {robust['Disponibilidad (%)'].mean():.2f} %", delta_color="off")
    fig_seed = go.Figure(go.Histogram(x=robust["MTBF (h)"], nbinsx=25, marker_color="#22d3ee"))
    fig_seed.add_vline(x=META_MTBF, line_dash="dash", line_color="#fb7185", annotation_text="Meta MTBF")
    fig_seed.add_vline(x=current_mtbf, line_color="#fbbf24", annotation_text=f"Semilla {int(seed)}", annotation_position="top left")
    fig_seed.update_layout(height=340, **THEME,
                           margin=dict(l=20, r=20, t=30, b=20), xaxis_title="MTBF predictivo medio de cada semilla (h)", yaxis_title="Semillas")
    st.plotly_chart(fig_seed, width="stretch", config=PLOTLY_CONFIG)
    st.caption("Cada barra agrupa semillas distintas. Si el valor esperado del MTBF queda cerca de la meta, que se cumpla o no "
               "depende de la semilla. Conviene reportar el promedio entre semillas y la probabilidad de cumplir, no un solo valor.")

    st.markdown('<div class="section-title">Sustento del supuesto de MTBF: ¿cuánto debe lograr la detección?</div>', unsafe_allow_html=True)
    st.caption("Un incremento del MTBF equivale a evitar una parte de las fallas no planificadas, convirtiéndolas en "
               "intervenciones programadas. Este panel compara esa parte con lo que logra la detección simulada con la "
               "configuración actual del panel lateral (ruido, pérdida de datos, umbral y método).")
    c_a, c_b = st.columns(2)
    p_prog = c_a.slider("Fallas con degradación progresiva (%)", 10, 100, step=5, key="p_prog",
                        help="Proporción de las fallas que avisan antes de ocurrir (descalibración, desgaste). "
                             "En el historial: 10 de 14 (71 %); ver la clasificación debajo.")
    min_lead_h = c_b.slider("Aviso mínimo para planificar (h)", 0.5, 8.0, step=0.5, key="min_lead_h",
                            help="Tiempo necesario para reservar el repuesto y programar la intervención en una pausa de turno.")
    with st.expander("Evidencia: clasificación de las 14 fallas del historial"):
        hist = pd.DataFrame(FALLAS_HISTORIAL, columns=["Equipo", "Falla registrada", "¿Progresiva?", "Variable que la anticiparía", "Medio de captura"])
        hist["Cobertura con la instrumentación base"] = COBERTURA_BASE
        st.dataframe(hist, hide_index=True, width="stretch")
        n_f = len(hist)
        n_si, n_par, n_no = (hist["¿Progresiva?"] == "Sí").sum(), (hist["¿Progresiva?"] == "Parcial").sum(), (hist["¿Progresiva?"] == "No").sum()
        medios = hist["Medio de captura"].value_counts()
        st.markdown(
            f"**Progresivas:** {n_si} de {n_f} ({n_si / n_f * 100:.1f} %); parciales: {n_par}; no anticipables con las variables propuestas: {n_no}.  \n"
            f"**Medio de captura:** sensor continuo {medios.get('Sensor continuo', 0)}, sensor e inspección {medios.get('Sensor e inspección', 0)}, "
            f"solo inspección TPM {medios.get('Inspección TPM', 0)}, sin cobertura {medios.get('Sin cobertura', 0)}.  \n"
            "**Por eje y componente:** CNC, 5 fallas en el eje X y 3 en el eje Y; torno, 2 en cabezal y caja de velocidades, 3 en carro, torreta y bancada, 1 en el contrapunto.  \n"
            f"**Cobertura con la instrumentación base** (un acelerómetro por equipo): plena {COBERTURA_BASE.count('Plena')}, parcial {COBERTURA_BASE.count('Parcial')}, "
            f"sin cobertura {COBERTURA_BASE.count('Sin cobertura')}. «Parcial» significa que la vibración de ese punto la ve un acelerómetro de contingencia, pero hay otra vía de detección.  \n"
            "**Recurrencias tras una intervención:** 2 (encoder del eje Y y holgura del carro), lo que sustenta recalibrar el gemelo después de cada intervención."
        )
        st.caption("Clasificación propuesta a partir de la descripción de cada falla; debe validarla el jefe de Mantenimiento. "
                   "La anticipación de las fallas que solo se detectan por inspección depende de la frecuencia de esa inspección, que la simulación aún no modela.")
    perf = detection_performance(sim, float(min_lead_h))
    f_required = 1 - 1 / (1 + mtbf_gain / 100)
    f_achievable = (p_prog / 100) * perf["p_anticipada"] if not np.isnan(perf["p_anticipada"]) else float("nan")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"Fallas a evitar para +{mtbf_gain} % de MTBF", f"{f_required * 100:.1f} %", delta_color="off")
    m2.metric(f"Fallas anticipadas con ≥ {min_lead_h:g} h de aviso", f"{perf['p_anticipada'] * 100:.0f} %",
              f"Aviso mediano: {perf['aviso_mediano_h']:.1f} h" if not np.isnan(perf["aviso_mediano_h"]) else "Sin detección", delta_color="off")
    m3.metric("Fallas evitables (progresivas × anticipadas)", f"{f_achievable * 100:.1f} %" if not np.isnan(f_achievable) else "—", delta_color="off")
    m4.metric("Falsas alarmas en operación normal", f"{perf['falsas_alarmas'] * 100:.0f} %", delta_color="off")
    if np.isnan(f_achievable):
        st.warning("Con esta duración y configuración no se alcanza la condición crítica sin intervenir, por lo que no hay fallas que anticipar. Aumenta la duración o usa otro escenario.")
    elif f_achievable >= f_required:
        st.success(f"El supuesto es coherente con la detección simulada: el gemelo podría evitar {f_achievable * 100:.1f} % de las fallas "
                   f"y se necesita {f_required * 100:.1f} %.")
    else:
        st.error(f"El supuesto no queda sustentado con esta configuración: el gemelo evitaría {f_achievable * 100:.1f} % de las fallas "
                 f"y se necesita {f_required * 100:.1f} %. Se requiere más fallas progresivas, más aviso o un umbral que anticipe más.")
    st.warning("Cota optimista: las señales son sintéticas y la operación normal simulada no incluye variación de carga, de velocidad ni de material. "
               "En planta, la anticipación será menor y habrá falsas alarmas; por eso el plan prevé seis semanas de captura pasiva para fijar el perfil normal por régimen.")
    st.caption(f"Resultado con {perf['fallas']} trayectorias que llegan a condición crítica sin intervenir ({'escenario Desgaste progresivo' if scenario == 'Operación normal' else 'escenario ' + scenario}). "
               "Aviso = horas entre la intervención propuesta y la condición crítica que se habría alcanzado. "
               "La reducción del MTTR es un supuesto aparte: depende de contar con el repuesto reservado (Componente 1) y de programar la intervención en una pausa de turno.")
    if detection == DETECCION_ESTADO:
        st.info("Con el método «estado interno» las falsas alarmas son 0 por construcción, porque el umbral se evalúa sobre el estado interno del simulador. "
                "Para medirlas con las lecturas, cambia el método de detección a «Estimado desde lecturas de sensores».")
    with st.expander("Compromiso del umbral de intervención (método por lecturas)"):
        st.caption("Un umbral más bajo da más aviso pero aumenta las falsas alarmas. Se calcula con 60 trayectorias por umbral, con el ruido y la pérdida de datos actuales.")
        if st.button("Calcular curva del umbral"):
            curve = threshold_tradeoff(sim, float(min_lead_h))
            fig_curve = go.Figure()
            fig_curve.add_trace(go.Scatter(x=curve["Umbral (%)"], y=curve["Fallas anticipadas (%)"], mode="lines+markers",
                                           name="Fallas anticipadas (%)", line=dict(color="#22d3ee")))
            fig_curve.add_trace(go.Scatter(x=curve["Umbral (%)"], y=curve["Falsas alarmas (%)"], mode="lines+markers",
                                           name="Falsas alarmas (%)", line=dict(color="#fb7185")))
            fig_curve.update_layout(height=340, **THEME,
                                    margin=dict(l=20, r=20, t=30, b=20), xaxis_title="Umbral de intervención (% de degradación)",
                                    yaxis_title="%", legend=dict(orientation="h", y=1.12))
            st.plotly_chart(fig_curve, width="stretch", config=PLOTLY_CONFIG)
            st.dataframe(curve.round(2), hide_index=True, width="stretch")


# ===========================================================================
# VISTA · ALERTS
# ===========================================================================

_log = get_logger()


def render_alerts(c):
    alerts, asset, asset_name, df, df_pr, df_re = c.alerts, c.asset, c.asset_name, c.df, c.df_pr, c.df_re
    idx_pr, intervention_index, duration, is_real, user = c.idx_pr, c.intervention_index, c.duration, c.is_real, c.user
    st.markdown('<div class="section-title">Registro de eventos del gemelo</div>', unsafe_allow_html=True)
    if alerts.empty:
        st.success("No se detectaron anomalías en el periodo analizado.")
    else:
        st.dataframe(alerts, width="stretch", hide_index=True)
        first_alert = alerts[~alerts["Nivel"].map(is_intervention)]
        if not first_alert.empty:
            alert_hour = float(first_alert.iloc[0]["Hora"])
            critical_rows = df[df["Estado"] == "Crítico"]
            lead_time = max(0.0, float(critical_rows.iloc[0]["Hora simulada"]) - alert_hour) if not critical_rows.empty else duration - alert_hour
            label = "Tiempo de anticipación" if is_real else "Tiempo de anticipación simulado"
            st.info(f"{label}: **{lead_time:.2f} horas** desde la primera alerta hasta la condición crítica o el final del periodo.")
        if intervention_index is not None and not is_real:
            st.success(f"El mantenimiento predictivo fue ejecutado virtualmente en la hora {df.iloc[intervention_index]['Hora simulada']:.2f}, "
                       "evitando que la degradación continuara hasta la falla.")

    title = "Orden de trabajo propuesta por el gemelo" + ("" if is_real else " (simulada)")
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if idx_pr is None:
        st.info("El gemelo no propone ninguna intervención: no se detectó degradación sostenida.")
        return

    before = max(idx_pr - 1, 0)
    deviations = {}
    for variable, cfg in asset["variables"].items():
        value = df_pr[variable].iloc[: before + 1].dropna()
        if value.empty:
            continue
        v = float(value.iloc[-1])
        span = (cfg["critical"] - cfg["base"]) if cfg["direction"] == "high" else (cfg["base"] - cfg["critical"])
        deviations[variable] = ((v - cfg["base"]) if cfg["direction"] == "high" else (cfg["base"] - v)) / span, v
    trigger = max(deviations, key=lambda k: deviations[k][0])
    t_int = float(df_pr["Hora simulada"].iloc[idx_pr])
    crit_rows = df_re.index[df_re["Estado"] == "Crítico"].tolist()
    lead = (float(df_re.loc[crit_rows[0], "Hora simulada"]) - t_int) if crit_rows else float("nan")
    if np.isnan(lead):
        lead_text = "No se alcanza la condición crítica en el periodo analizado"
    elif lead < 0:
        lead_text = "La condición crítica ya se había alcanzado"
    else:
        lead_text = f"{lead:.2f} h"
    ot_id = f"OT-{asset['code']}-{c.data_id}-{t_int:05.2f}" if is_real else f"OT-SIM-{asset['code']}-{t_int:05.2f}"
    source = "datos reales" if is_real else "simulación"

    left_ot, right_ot = st.columns([1.6, 1])
    with left_ot:
        order = pd.DataFrame({
            "Campo": ["Orden", "Equipo", "Variable que la origina", "Lectura / umbral de advertencia", "Hora de la propuesta",
                      "Aviso antes de la condición crítica", "Ventana sugerida", "Repuestos", "Origen"],
            "Valor": [ot_id, f"{asset_name} · {asset['code']}", trigger,
                      f"{value_text(trigger, deviations[trigger][1])} / {asset['variables'][trigger]['warning']}", f"{t_int:.2f} h",
                      lead_text, "Próxima pausa de turno", "Por definir por el jefe de Mantenimiento",
                      "Gemelo digital (propuesta consultiva)"],
        })
        st.dataframe(order, hide_index=True, width="stretch")
    with right_ot:
        st.markdown("**Decisión del jefe de Mantenimiento**")
        previous = last_decision(DB_PATH, ot_id)
        if previous:
            icon = "✅" if previous["decision"] == "Aprobar" else "⛔"
            st.info(f"{icon} {previous['decision']} · {previous['username']} · {previous['ts'].replace('T', ' ')}")
        else:
            st.info("Pendiente de revisión. El gemelo es consultivo: no envía comandos a la máquina.")
        if can(user, "decidir"):
            choice = st.radio("Nueva decisión", ["Aprobar", "Rechazar"], index=None, horizontal=True, key=f"decision_{ot_id}",
                              label_visibility="collapsed")
            if st.button("Registrar decisión", disabled=choice is None, key=f"save_{ot_id}"):
                record_decision(DB_PATH, ot_id, asset_name, choice, user["username"], source)
                _log.info("Decisión %s sobre %s registrada por %s", choice, ot_id, user["username"])
                st.rerun()
        else:
            st.caption("Tu rol solo permite consultar. La decisión la registra el jefe de Mantenimiento.")


# ===========================================================================
# VISTA · TESTS
# ===========================================================================

def render_tests(c):
    (asset, completeness, experiments) = (c.asset, c.completeness, experiment_results(c.sim))
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
        st.success("La calidad de datos cumple el criterio propuesto de ≥95 %.")


# ===========================================================================
# VISTA · IMPLEMENTATION
# ===========================================================================

def render_implementation(c):
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


# ===========================================================================
# VISTA · METHODOLOGY
# ===========================================================================

def render_methodology(c):
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
        - El error de posición del CNC se simula cada 5 minutos en el modo por defecto; con la captura «por inspección TPM» solo existe en cada inspección, como ocurre en planta (el controlador no lo exporta), y se excluye de la calidad de datos de sensores.
        - En el método «estimado desde lecturas», el gemelo aprende el perfil normal con las primeras 4 horas y lo recalibra con las 4 horas posteriores a cada intervención; con el método de estado interno usa las constantes del diagnóstico.
        - La variación de la carga operativa mueve las lecturas sin que haya degradación; con la compensación de régimen el gemelo descuenta la carga conocida, salvo un 20 % de error.
        - El incremento del MTBF y la reducción del MTTR son supuestos configurables. La simulación no los calcula a partir de la detección.
        - En el método "estado interno", el cruce del umbral se evalúa sobre el estado interno del simulador y se agregan 4 intervalos de espera. El método "estimado desde lecturas" usa solo las lecturas y exige 4 seguidas por encima del umbral.
        - La dispersión del Monte Carlo (formas gamma 9 y 8) es un supuesto del modelo.
        - Los umbrales de vibración, temperatura y corriente son demostrativos y deben calibrarse con mediciones normales del equipo, especificaciones del fabricante y evaluación técnica.
        - La reducción del MTTR y la proporción de fallas con degradación progresiva son supuestos; el panel «Sustento del supuesto» muestra cuánto debe lograr la detección para sostener el incremento del MTBF.
        - El aviso de la detección (horas entre la intervención propuesta y la condición crítica) depende del umbral y del ruido; con el umbral de 66 % puede ser de pocas horas.
        - El modelo 3D es esquemático y no está a escala.
        """
    )


# ===========================================================================
# VISTA · HISTORY
# ===========================================================================

def render_history(c):
    st.markdown('<div class="section-title">Historial de decisiones</div>', unsafe_allow_html=True)
    st.caption("Cada decisión del jefe de Mantenimiento queda guardada con su fecha y usuario, aunque se cierre la aplicación.")
    decisions = list_decisions(DB_PATH)
    if decisions.empty:
        st.info("Aún no hay decisiones registradas. Aparecerán aquí cuando se apruebe o rechace una orden en la vista Alertas.")
    else:
        st.dataframe(decisions, width="stretch", hide_index=True)
        st.download_button("⬇ Descargar decisiones (CSV)", data=decisions.to_csv(index=False).encode("utf-8"),
                           file_name="decisiones.csv", mime="text/csv")

    st.markdown('<div class="section-title">Intervenciones propuestas con datos reales</div>', unsafe_allow_html=True)
    events = list_events(DB_PATH)
    if events.empty:
        st.info("Aún no hay intervenciones propuestas. Se registran automáticamente al analizar lecturas reales.")
    else:
        equipment = st.selectbox("Equipo", ["Todos"] + sorted(events["Equipo"].unique()), key="history_asset")
        shown = events if equipment == "Todos" else events[events["Equipo"] == equipment]
        st.dataframe(shown, width="stretch", hide_index=True)
        st.download_button("⬇ Descargar eventos (CSV)", data=shown.to_csv(index=False).encode("utf-8"),
                           file_name="eventos.csv", mime="text/csv")


# ===========================================================================
# VISTA · ADMIN
# ===========================================================================

_log = get_logger()


def _thresholds_editor():
    st.markdown('<div class="section-title">Umbrales por equipo</div>', unsafe_allow_html=True)
    st.caption("Los valores por defecto son demostrativos: deben calibrarse con la línea base real y las especificaciones del fabricante. "
               f"Se guardan en `{CONFIG_PATH}` y se aplican de inmediato.")
    asset_name = st.selectbox("Equipo", list(ASSETS.keys()), key="admin_asset")
    variables = ASSETS[asset_name]["variables"]
    table = pd.DataFrame({
        "Variable": list(variables),
        "Sentido": ["sube al degradarse" if v["direction"] == "high" else "baja al degradarse" for v in variables.values()],
        "Valor normal": [v["base"] for v in variables.values()],
        "Advertencia": [v["warning"] for v in variables.values()],
        "Crítico": [v["critical"] for v in variables.values()],
    })
    edited = st.data_editor(table, hide_index=True, width="stretch", key=f"thr_{asset_name}",
                            disabled=["Variable", "Sentido", "Valor normal"])
    left, right = st.columns(2)
    if left.button("Guardar umbrales", key="thr_save"):
        errors, new_values = [], {}
        for _, row in edited.iterrows():
            spec = variables[row["Variable"]]
            problems = validate_thresholds(spec["direction"], spec["base"], float(row["Advertencia"]), float(row["Crítico"]))
            if problems:
                errors.append(f"{row['Variable']}: {problems[0]}")
            new_values[row["Variable"]] = {"warning": float(row["Advertencia"]), "critical": float(row["Crítico"])}
        if errors:
            st.error("No se guardó:\n\n" + "\n".join(f"- {e}" for e in errors))
        else:
            overrides = load_overrides(CONFIG_PATH)
            overrides[asset_name] = new_values
            save_overrides(CONFIG_PATH, overrides)
            st.cache_data.clear()
            _log.info("Umbrales de %s actualizados", asset_name)
            st.rerun()
    if right.button("Restaurar valores de fábrica", key="thr_reset"):
        overrides = load_overrides(CONFIG_PATH)
        overrides.pop(asset_name, None)
        save_overrides(CONFIG_PATH, overrides)
        st.cache_data.clear()
        _log.info("Umbrales de %s restaurados", asset_name)
        st.rerun()


def _integrations():
    st.markdown('<div class="section-title">Notificaciones</div>', unsafe_allow_html=True)
    channels = channels_configured()
    st.write(f"Correo: **{'configurado' if channels['correo'] else 'sin configurar'}** · "
             f"Webhook: **{'configurado' if channels['webhook'] else 'sin configurar'}**")
    st.caption("Se configuran con variables de entorno (ver README). Se avisa una sola vez por cada intervención propuesta con datos reales.")
    if st.button("Enviar notificación de prueba", disabled=not any(channels.values()), key="notify_test"):
        sent = notify("Gemelo digital: prueba de notificación", "Si recibes este mensaje, el canal está bien configurado.")
        st.success(f"Enviada por: {', '.join(sent)}") if sent else st.error("No se pudo enviar. Revisa el registro de la aplicación.")


def _users_and_data():
    st.markdown('<div class="section-title">Usuarios y datos</div>', unsafe_allow_html=True)
    users = load_users(USERS_PATH)
    if users:
        st.dataframe(pd.DataFrame([{"Usuario": u, "Nombre": r.get("name", u), "Rol": ROLE_LABEL[r["role"]]} for u, r in users.items()]),
                     hide_index=True, width="stretch")
    else:
        st.warning(f"No hay archivo de usuarios (`{USERS_PATH}`): la aplicación funciona en modo demostración, sin inicio de sesión. "
                   "Para producción, crea usuarios con `python crear_usuario.py`.")
    st.caption(f"Base de datos: `{DB_PATH}` · {len(list_decisions(DB_PATH))} decisiones · {len(list_events(DB_PATH))} eventos · registros en `{DATA_DIR}`.")


def render_admin(c):
    _thresholds_editor()
    _integrations()
    _users_and_data()


# ===========================================================================
# VISTA · ANALYSIS
# ===========================================================================

def render_analysis(c):
    if c.is_real:
        st.info("Este análisis usa el simulador con los controles de la barra lateral; no depende del archivo de datos reales cargado.")
    tab_sens, tab_tests = st.tabs(["📈 Sensibilidad", "🧪 Pruebas"])
    with tab_sens:
        render_sensitivity(c)
    with tab_tests:
        render_tests(c)


# ===========================================================================
# VISTA · HELP
# ===========================================================================

def render_help(c):
    tab_use, tab_method, tab_impl = st.tabs(["Cómo usar", "Metodología", "Implementación"])
    with tab_use:
        st.markdown(
            "**Operación diaria**\n"
            "1. En la barra lateral elige el equipo y la fuente de datos (simulación o CSV de lecturas reales).\n"
            "2. **Telemetría** y **Réplica virtual** muestran el estado del equipo en el instante elegido.\n"
            "3. **Alertas** muestra las advertencias y la orden de trabajo que propone el gemelo. El jefe de Mantenimiento la aprueba o la rechaza.\n"
            "4. **Historial** conserva cada decisión con su fecha y usuario.\n"
            "5. **Indicadores** resume disponibilidad, MTBF y MTTR frente a las metas.\n\n"
            "**Datos reales:** el CSV necesita una columna de tiempo y una columna por variable del equipo. "
            "Descarga la plantilla desde la barra lateral. Las primeras 4 horas se usan para aprender el perfil normal del equipo, "
            "así que deben corresponder a operación sin fallas.\n\n"
            "**Importante:** el gemelo es consultivo. No envía comandos a la máquina."
        )
        st.caption(f"Versión {APP_VERSION}")
    with tab_method:
        render_methodology(c)
    with tab_impl:
        render_implementation(c)


# ===========================================================================
# APLICACIÓN
# ===========================================================================

RENDERERS = {
    V_TELEMETRIA: render_telemetry, V_REPLICA: render_replica, V_ALERTAS: render_alerts, V_HISTORIAL: render_history,
    V_INDICADORES: render_indicators, V_ANALISIS: render_analysis, V_CONFIG: render_admin, V_AYUDA: render_help,
}
# Controles que viven dentro de una vista: se conservan al cambiar de vista (Streamlit descarta
# el estado de los widgets que no se dibujan en una ejecución).
VIEW_KEYS = ("p_prog", "min_lead_h", "history_asset", "admin_asset")
MAX_LOGIN_ATTEMPTS = 5

st.set_page_config(page_title=APP_TITLE, page_icon="⚙️", layout="wide", initial_sidebar_state="expanded")
inject_css()
_log = get_logger()


# ---------------------------------------------------------------------------
# Acceso
# ---------------------------------------------------------------------------
def current_user() -> dict:
    """Usuario con sesión iniciada. Sin archivo de usuarios, modo demostración (sin inicio de sesión)."""
    users = load_users(USERS_PATH)
    if not users:
        return DEMO_USER
    user = st.session_state.get("user")
    if user:
        return user
    _, center, _ = st.columns([1, 1.2, 1])
    with center:
        st.markdown(f"## ⚙️ {APP_TITLE}")
        attempts = st.session_state.get("login_attempts", 0)
        if attempts >= MAX_LOGIN_ATTEMPTS:
            st.error("Demasiados intentos fallidos. Cierra y vuelve a abrir la pestaña para intentar de nuevo.")
            st.stop()
        with st.form("login"):
            username = st.text_input("Usuario")
            password = st.text_input("Contraseña", type="password")
            submitted = st.form_submit_button("Ingresar", width="stretch")
        if submitted:
            found = authenticate(users, username, password)
            if found:
                st.session_state["user"] = found
                st.session_state["login_attempts"] = 0
                _log.info("Inicio de sesión: %s (%s)", found["username"], found["role"])
                st.rerun()
            st.session_state["login_attempts"] = attempts + 1
            _log.warning("Intento de inicio de sesión fallido para el usuario %r", username)
            time.sleep(1)
            st.error("Usuario o contraseña incorrectos.")
    st.stop()


def render_user_box(user: dict, auth_enabled: bool) -> None:
    with st.sidebar:
        if auth_enabled:
            st.caption(f"👤 {user['name']} · {ROLE_LABEL[user['role']]}")
            if st.button("Cerrar sesión", width="stretch"):
                st.session_state.pop("user", None)
                st.rerun()
        else:
            st.caption("👤 Modo demostración (sin inicio de sesión)")


def keep_view_state() -> None:
    for key in list(st.session_state.keys()):
        if key in VIEW_KEYS or key.startswith("decision_"):
            st.session_state[key] = st.session_state[key]


# ---------------------------------------------------------------------------
# Datos del análisis
# ---------------------------------------------------------------------------
def build_context(ctl, user) -> SimpleNamespace:
    """Calcula una sola vez los resultados que comparten todas las vistas."""
    is_real = ctl.data_source == DATOS_REALES
    sim = SimConfig(
        asset_name=ctl.asset_name, scenario=ctl.scenario, duration=ctl.duration, seed=ctl.seed,
        noise_factor=ctl.noise_factor, missing_rate=ctl.missing_rate,
        intervention_threshold=ctl.intervention_threshold, detection=ctl.detection,
        load_variability=ctl.load_pct / 100, compensate_load=ctl.compensate_load,
        inspection_h=ctl.inspection_eff, failure_mode=ctl.failure_mode, thresholds_id=fingerprint(ASSETS),
    )
    data_id, duration, inspection_eff = "", float(ctl.duration), ctl.inspection_eff
    if is_real:
        if ctl.csv_bytes is None:
            st.info("Carga un CSV de lecturas en la barra lateral para analizar datos reales. Puedes descargar una plantilla con el formato esperado.")
            st.stop()
        try:
            df, alerts_df, idx = analyze_csv(ctl.csv_bytes, ctl.asset_name, ctl.intervention_threshold, sim.thresholds_id)
        except IngestError as error:
            st.error(str(error))
            st.stop()
        data_id = data_fingerprint(ctl.csv_bytes)
        df_re = df_pr = df
        alerts_re = alerts_pr = alerts_df
        idx_re, idx_pr = None, idx
        intervention_index = idx
        duration, inspection_eff = float(df["Hora simulada"].iloc[-1]), 0.0
        register_proposal(ctl.asset_name, data_id, df, alerts_df, idx)
    else:
        df_re, alerts_re, idx_re = simulate_telemetry(sim, "Reactivo")
        df_pr, alerts_pr, idx_pr = simulate_telemetry(sim, "Predictivo")
        df, alerts_df, intervention_index = (df_re, alerts_re, idx_re) if ctl.policy == "Reactivo" else (df_pr, alerts_pr, idx_pr)

    asset = ASSETS[ctl.asset_name]
    operating, base_mtbf, base_mttr, base_availability = base_metrics(asset)
    det_mtbf = base_mtbf * (1 + ctl.mtbf_gain / 100)
    det_mttr = base_mttr * (1 - ctl.mttr_reduction / 100)
    det_availability = availability_pct(det_mtbf, det_mttr)
    projected_mtbf, projected_mttr = (det_mtbf, det_mttr) if (ctl.policy == "Predictivo" or is_real) else (base_mtbf, base_mttr)
    projected_availability = availability_pct(projected_mtbf, projected_mttr)
    projected_downtime = asset["scheduled_hours"] * (1 - projected_availability / 100)

    return SimpleNamespace(
        user=user, is_real=is_real, data_id=data_id, sim=sim, asset=asset,
        df_re=df_re, alerts_re=alerts_re, idx_re=idx_re, df_pr=df_pr, alerts_pr=alerts_pr, idx_pr=idx_pr,
        df=df, alerts=alerts_df, intervention_index=intervention_index,
        operating=operating, base_mtbf=base_mtbf, base_mttr=base_mttr, base_availability=base_availability,
        det_mtbf=det_mtbf, det_mttr=det_mttr, det_availability=det_availability,
        projected_mtbf=projected_mtbf, projected_mttr=projected_mttr, projected_availability=projected_availability,
        projected_downtime=projected_downtime,
        avoided_downtime=max(0.0, asset["downtime_hours"] - projected_downtime),
        completeness=data_quality(df, ctl.asset_name, inspection_eff > 0), inspection_eff=inspection_eff,
        asset_name=ctl.asset_name, scenario=ctl.scenario, policy=ctl.policy, duration=duration, seed=ctl.seed,
        mtbf_gain=ctl.mtbf_gain, mttr_reduction=ctl.mttr_reduction, simulations=ctl.simulations, detection=ctl.detection,
    )


# ---------------------------------------------------------------------------
# Encabezado
# ---------------------------------------------------------------------------
def render_header(c) -> None:
    subtitle = "con lecturas reales de sensores" if c.is_real else "con telemetría simulada"
    st.markdown(
        f"""
        <div class="hero">
          <div class="eyebrow">Industria 4.0 · TPM · Gemelos digitales consultivos</div>
          <h1>Gemelos digitales de mantenimiento</h1>
          <p>Un gemelo digital por equipo (CNC plasma y torno), {subtitle}, detección de degradación y evaluación de la política de mantenimiento.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    hour_selected = st.slider(
        "Instante a revisar (horas desde el inicio)", 0.0, float(c.duration), float(c.duration),
        step=1 / SAMPLES_PER_HOUR, format="%.2f h",
    )
    c.moment = int(min(max(round(hour_selected * SAMPLES_PER_HOUR), 0), len(c.df) - 1))
    c.hour_now = float(c.df.iloc[c.moment]["Hora simulada"])
    st.plotly_chart(timeline_figure(c.df_re, c.df_pr, c.alerts_re, c.alerts_pr, c.idx_pr, c.hour_now),
                    width="stretch", config=PLOTLY_CONFIG_BARE)
    st.caption("Franja de estado por política: verde = normal, amarillo = advertencia, rojo = crítico. ★ primera alerta; "
               "◆ intervención propuesta por el gemelo."
               + (" Con datos reales ambas franjas muestran las mismas lecturas." if c.is_real else
                  " Cambia al mover el escenario, el ruido, la pérdida de datos, el umbral o el método de detección."))

    c.status_now = str(c.df.iloc[c.moment]["Estado"])
    c.health_now = float(min(100.0, max(0.0, 100 - float(c.df.iloc[c.moment]["Degradación"]))))
    assumption = "Calculado con los supuestos de mejora de la barra lateral; no es una medición."
    col_status, col_health, col_risk, col_data = st.columns([1.6, 1, 1, 1])
    with col_status:
        status_card(c.asset_name, c.status_now, c.health_now, c.hour_now, live=c.is_real)
    with col_health:
        st.metric("Disponibilidad proyectada", f"{c.projected_availability:.2f}%", f"{c.projected_availability - c.base_availability:+.2f} pp", help=assumption)
        st.metric("MTBF proyectado", f"{c.projected_mtbf:.2f} h", f"{c.projected_mtbf - c.base_mtbf:+.2f} h", help=assumption)
    with col_risk:
        st.metric("MTTR proyectado", f"{c.projected_mttr:.2f} h", f"{c.projected_mttr - c.base_mttr:+.2f} h", delta_color="inverse", help=assumption)
        st.metric("Parada potencial evitada", f"{c.avoided_downtime:.2f} h", "Periodo diagnóstico", delta_color="off", help=assumption)
    with col_data:
        st.metric("Calidad de datos", f"{c.completeness:.2f}%", f"Meta ≥ {META_CALIDAD_DATOS:.0f}%", delta_color="off")
        st.metric("Registros" + ("" if c.is_real else " simulados"), f"{len(c.df):,}", "Cada 5 min", delta_color="off")

    # lecturas en el instante elegido (última lectura válida si el dato se perdió)
    c.readings = {v: last_valid(c.df[v], c.moment) for v in c.asset["variables"]}


def render_config_download(ctl) -> None:
    export = {
        "version_app": APP_VERSION,
        "generado": datetime.now().isoformat(timespec="seconds"),
        "fuente_datos": ctl.data_source, "activo": ctl.asset_name, "escenario": ctl.scenario, "politica": ctl.policy,
        "duracion_h": ctl.duration, "semilla": ctl.seed,
        "incremento_mtbf_pct": ctl.mtbf_gain, "reduccion_mttr_pct": ctl.mttr_reduction,
        "corridas_monte_carlo": ctl.simulations, "ruido_sensor": ctl.noise_factor,
        "perdida_datos_pct": ctl.missing_rate_pct, "umbral_intervencion_pct": ctl.intervention_pct,
        "metodo_deteccion": ctl.detection, "captura_error_posicion": ctl.position_capture,
        "intervalo_inspeccion_h": ctl.inspection_eff, "variacion_carga_pct": ctl.load_pct,
        "compensar_regimen": ctl.compensate_load, "modo_falla": ctl.failure_mode,
    }
    with st.sidebar:
        st.download_button(
            "⬇ Descargar configuración (JSON)",
            data=json.dumps(export, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="configuracion_simulacion.json", mime="application/json", width="stretch",
        )


def main() -> None:
    for message in reload_thresholds(ASSETS, ASSETS_FACTORY, CONFIG_PATH):
        _log.debug("Umbral: %s", message)
    auth_enabled = bool(load_users(USERS_PATH))
    user = current_user()
    keep_view_state()
    render_user_box(user, auth_enabled)
    controls = render_sidebar()
    ctx = build_context(controls, user)
    render_header(ctx)

    views = [v for v in VIEWS if v != V_CONFIG or can(user, "configurar")]
    if st.session_state.get("view") not in views:
        st.session_state.pop("view", None)
    view = st.radio("Vista", views, horizontal=True, label_visibility="collapsed", key="view")
    RENDERERS[view](ctx)       # solo se calcula y dibuja la vista activa
    render_config_download(controls)
    st.markdown("---")
    st.caption(f"Gemelo digital consultivo · Fuente: {'datos reales' if ctx.is_real else 'datos simulados'} · v{APP_VERSION}")


main()
