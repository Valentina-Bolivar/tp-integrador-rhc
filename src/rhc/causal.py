"""Inferencia causal (Fase 4): DAG, criterio backdoor, IPW, sensibilidad y CATE.

Convenciones: T = rhc (0/1), Y = dth30 (0/1), C = los (días de internación, colisionador),
Z = covariables basales estandarizadas (conjunto de ajuste). Los efectos se expresan como
diferencias de riesgo: un ATE de 0,05 equivale a 5 puntos porcentuales de mortalidad.
"""

from __future__ import annotations

import re

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from rhc.config import OUTCOME_BIN, SEED, TREATMENT
from rhc.features import build_preprocessor

COLLIDER = "los"
MEDIATOR = "intensificacion_terapeutica"  # nodo no observado del DAG

# DAG conceptual por bloques: nodos agregados para la figura, el informe y la app.
DAG_NODES = {
    "Z1": "Demografía y nivel socioeconómico",
    "Z2": "Diagnóstico de ingreso y comorbilidades",
    "Z3": "Gravedad fisiológica del día 1",
    "Z4": "Pronóstico y preferencias (sobrevida estimada, DASI, no reanimar)",
    "T": "Cateterismo cardíaco derecho (RHC)",
    "M": "Intensificación terapéutica (no observada)",
    "Y": "Muerte a 30 días",
    "C": "Días de internación (colisionador)",
    "U": "Criterio médico y hospital (no observados)",
}
DAG_EDGES = [
    ("Z1", "Z2"), ("Z1", "Z4"), ("Z2", "Z3"), ("Z2", "Z4"), ("Z3", "Z4"),
    ("Z1", "T"), ("Z2", "T"), ("Z3", "T"), ("Z4", "T"),
    ("Z1", "Y"), ("Z2", "Y"), ("Z3", "Y"), ("Z4", "Y"),
    ("T", "M"), ("M", "Y"), ("T", "Y"),
    ("T", "C"), ("Y", "C"),
    ("U", "T"), ("U", "Y"),
]
LATENT = {"M", "U"}


def safe_name(name: str) -> str:
    """Nombre de columna compatible con las fórmulas de DoWhy."""
    return re.sub(r"[^0-9a-zA-Z_]", "_", name)


def causal_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Covariables basales estandarizadas (Z) junto con T, Y y el colisionador.

    El análisis causal usa la muestra completa: no hay evaluación predictiva fuera de
    muestra que proteger, y la precisión del ATE crece con n. Se excluye el único paciente
    sin duración de internación para poder ilustrar el sesgo de colisionador.
    """
    pre = build_preprocessor(scale=True, include_treatment=False)
    Z = pre.fit_transform(df).astype(float)
    Z.columns = [safe_name(c) for c in Z.columns]
    data = pd.concat([Z, df[[TREATMENT, OUTCOME_BIN, COLLIDER]]], axis=1)
    data = data.dropna(subset=[COLLIDER])
    return data, list(Z.columns)


def dowhy_graph(confounders: list[str]) -> nx.DiGraph:
    """DAG operativo para DoWhy: cada confusor apunta a T y a Y; T causa Y directamente
    y a través del mediador no observado; la estadía es colisionador entre T e Y."""
    g = nx.DiGraph()
    g.add_edges_from((z, TREATMENT) for z in confounders)
    g.add_edges_from((z, OUTCOME_BIN) for z in confounders)
    g.add_edges_from(
        [
            (TREATMENT, OUTCOME_BIN),
            (TREATMENT, MEDIATOR),
            (MEDIATOR, OUTCOME_BIN),
            (TREATMENT, COLLIDER),
            (OUTCOME_BIN, COLLIDER),
        ]
    )
    return g


def dag_dot() -> str:
    """DAG conceptual en formato DOT (las líneas punteadas marcan variables latentes)."""
    lines = ["digraph RHC {", "  rankdir=LR;"]
    for key, label in DAG_NODES.items():
        style = ', style="dashed"' if key in LATENT else ""
        lines.append(f'  {key} [label="{label}"{style}];')
    for a, b in DAG_EDGES:
        style = ' [style="dashed"]' if a in LATENT or b in LATENT else ""
        lines.append(f"  {a} -> {b}{style};")
    lines.append("}")
    return "\n".join(lines)


def propensity_scores(Z: pd.DataFrame, t: np.ndarray) -> np.ndarray:
    """P(T = 1 | Z) estimada por regresión logística sobre las covariables estandarizadas."""
    return LogisticRegression(max_iter=5000).fit(Z, t).predict_proba(Z)[:, 1]


def hajek_ipw(y, t, ps, trim: tuple[float, float] | None = None) -> dict:
    """Estimador IPW normalizado (Hájek) de los resultados potenciales y del ATE."""
    y, t, ps = map(np.asarray, (y, t, ps))
    keep = np.ones_like(t, dtype=bool) if trim is None else (ps > trim[0]) & (ps < trim[1])
    y, t, ps = y[keep], t[keep], ps[keep]
    w1, w0 = t / ps, (1 - t) / (1 - ps)
    mu1 = np.sum(w1 * y) / np.sum(w1)
    mu0 = np.sum(w0 * y) / np.sum(w0)
    return {"mu1": mu1, "mu0": mu0, "ATE": mu1 - mu0, "n": int(keep.sum())}


def standardized_differences(Z: pd.DataFrame, t: np.ndarray, weights=None) -> pd.Series:
    """Diferencia de medias estandarizada por covariable, con o sin ponderación IPW."""
    w = np.ones(len(t)) if weights is None else np.asarray(weights)
    t = np.asarray(t)
    pooled_sd = np.sqrt((Z[t == 1].var() + Z[t == 0].var()) / 2)
    m1 = Z[t == 1].mul(w[t == 1], axis=0).sum() / w[t == 1].sum()
    m0 = Z[t == 0].mul(w[t == 0], axis=0).sum() / w[t == 0].sum()
    return (m1 - m0) / pooled_sd


def e_value(rr: float) -> float:
    """E-value (VanderWeele y Ding, 2017): fuerza mínima de asociación, en escala de riesgo
    relativo, que un confusor no observado debería tener con T y con Y para anular el efecto."""
    rr = rr if rr >= 1 else 1 / rr
    return rr + np.sqrt(rr * (rr - 1))


def simulate_trap_questions(n: int = 200_000, seed: int = SEED) -> pd.DataFrame:
    """Simulación con efecto conocido: qué estima la regresión según el conjunto de ajuste.

    DAG simulado: Z → T, Z → Y, T → M → Y, T → Y y T → C ← Y.
    Efecto total verdadero = directo (0,04) + indirecto (0,5 × 0,06 = 0,03) = 0,07.
    """
    import statsmodels.api as sm

    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t = rng.binomial(1, 1 / (1 + np.exp(-(0.8 * z))))
    m = 0.5 * t + rng.normal(size=n)
    y = 0.30 + 0.04 * t + 0.06 * m + 0.10 * z + rng.normal(0, 0.3, size=n)
    c = 0.8 * t - 1.5 * y + rng.normal(size=n)
    sets = {
        "Sin ajuste (ingenuo)": [],
        "Z (criterio backdoor)": [z],
        "Z + mediador M": [z, m],
        "Z + colisionador C": [z, c],
    }
    rows = []
    for name, extra in sets.items():
        X = sm.add_constant(np.column_stack([t, *extra]) if extra else t)
        beta_t = sm.OLS(y, X).fit().params[1]
        rows.append({"conjunto de ajuste": name, "efecto estimado de T": beta_t})
    out = pd.DataFrame(rows).set_index("conjunto de ajuste")
    out["efecto total verdadero"] = 0.07
    out["sesgo"] = out["efecto estimado de T"] - 0.07
    return out
