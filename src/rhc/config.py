"""Configuración central del proyecto: rutas, semilla y grupos de variables.

Todas las fases importan desde aquí para garantizar que el notebook, la app y los
tests usen exactamente las mismas definiciones (contrato de datos entre fases).
"""

from __future__ import annotations

from pathlib import Path

# --- Rutas ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw" / "rhc.csv"
DATA_URL = "https://hbiostat.org/data/repo/rhc.csv"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"

# --- Reproducibilidad -------------------------------------------------------
SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
ALPHA = 0.05

# --- Roles causales (contrato de datos de la Fase 4) ------------------------
TREATMENT = "rhc"  # 1 = se aplicó cateterismo cardíaco derecho en las primeras 24 h
OUTCOME_BIN = "dth30"  # 1 = muerte dentro de los 30 días (resultado de Connors et al., 1996)
OUTCOME_BIN_180 = "death"  # 1 = muerte dentro de los 180 días (análisis de sensibilidad)
OUTCOME_CONT = "log_los"  # log(días de internación), objetivo del MCO de la Fase 2

# --- Grupos de variables ------------------------------------------------------
COMORBIDITIES = [
    "cardiohx", "chfhx", "dementhx", "psychhx", "chrpulhx", "renalhx",
    "liverhx", "gibledhx", "malighx", "immunhx", "transhx", "amihx",
]
ADMISSION_DX = [
    "resp", "card", "neuro", "gastr", "renal", "meta",
    "hema", "seps", "trauma", "ortho",
]
PHYSIOLOGY = [
    "meanbp1", "wblc1", "hrt1", "resp1", "temp1", "pafi1", "alb1",
    "hema1", "bili1", "crea1", "sod1", "pot1", "paco21", "ph1", "wtkilo1",
]
SCORES = ["aps1", "scoma1", "surv2md1", "das2d3pc"]
DEMOGRAPHICS_NUM = ["age", "edu"]
CATEGORICAL = ["cat1", "cat2", "ca", "sex", "race", "income", "ninsclas", "dnr1"]

# Variables posteriores al tratamiento o con definición de resultado: nunca se usan
# como predictoras ni como confusoras (evita fuga de datos y sesgo post-tratamiento).
POST_TREATMENT = [
    "sadmdte", "dschdte", "dthdte", "lstctdte", "t3d30",
    "death", "dth30", "los", "log_los",
]
ID_COLS = ["Unnamed: 0", "ptid"]

NUMERIC_FEATURES = DEMOGRAPHICS_NUM + SCORES + PHYSIOLOGY
BINARY_FEATURES = COMORBIDITIES + ADMISSION_DX

# Etiquetas en español para gráficos y tablas.
LABELS = {
    "rhc": "Cateterismo cardíaco derecho (RHC)",
    "dth30": "Muerte a 30 días",
    "death": "Muerte a 180 días",
    "los": "Días de internación",
    "log_los": "log(días de internación)",
    "age": "Edad (años)",
    "edu": "Educación (años)",
    "aps1": "Score APACHE III",
    "scoma1": "Escala de coma (SUPPORT)",
    "surv2md1": "Prob. de sobrevida a 2 meses (modelo SUPPORT)",
    "das2d3pc": "Índice DASI (capacidad funcional)",
    "meanbp1": "Presión arterial media (mmHg)",
    "wblc1": "Leucocitos (×10⁹/L)",
    "hrt1": "Frecuencia cardíaca (lpm)",
    "resp1": "Frecuencia respiratoria (rpm)",
    "temp1": "Temperatura (°C)",
    "pafi1": "PaO₂/FiO₂",
    "alb1": "Albúmina (g/dL)",
    "hema1": "Hematocrito (%)",
    "bili1": "Bilirrubina (mg/dL)",
    "crea1": "Creatinina (mg/dL)",
    "sod1": "Sodio (mEq/L)",
    "pot1": "Potasio (mEq/L)",
    "paco21": "PaCO₂ (mmHg)",
    "ph1": "pH arterial",
    "wtkilo1": "Peso (kg)",
    "cat1": "Categoría diagnóstica primaria",
    "cat2": "Categoría diagnóstica secundaria",
    "ca": "Cáncer",
    "income": "Ingreso anual",
    "ninsclas": "Cobertura médica",
    "dnr1": "Orden de no reanimar (día 1)",
}
