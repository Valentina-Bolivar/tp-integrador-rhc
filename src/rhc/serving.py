"""Capa de servicio de la aplicación web: carga de artefactos, predicción y consultas causales.

La aplicación no reestima los modelos del trabajo: lee los artefactos que el notebook guarda
en ``models/`` (modelo logístico, DAG y resultados causales). Si el notebook se vuelve a
ejecutar, la aplicación refleja automáticamente los resultados nuevos.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from rhc import config as cfg
from rhc.causal import causal_frame, propensity_scores, safe_name, standardized_differences
from rhc.cleaning import apply_sentinels
from rhc.data import load_frame
from rhc.features import build_pca, build_preprocessor, split
from rhc.inference import wilson_ci

N_BOOT = 300


@dataclass
class Artifacts:
    data: pd.DataFrame
    train: pd.DataFrame
    test: pd.DataFrame
    audit: pd.DataFrame
    model: object
    model_meta: dict
    causal: dict
    dag: dict
    pca: object
    boot_coefs: np.ndarray
    test_proba: np.ndarray
    balance: pd.DataFrame
    propensity: pd.DataFrame


def load_artifacts() -> Artifacts:
    """Carga datos, modelos y resultados. Pensada para cachearse una vez por servidor."""
    data, audit = apply_sentinels(load_frame())
    train, test = split(data)
    model = joblib.load(cfg.MODELS_DIR / "modelo_logistico.joblib")
    meta = json.loads((cfg.MODELS_DIR / "modelo_logistico.json").read_text())
    causal = json.loads((cfg.MODELS_DIR / "resultados_causales.json").read_text())
    dag = json.loads((cfg.MODELS_DIR / "dag.json").read_text())
    pca = build_pca(n_components=2).fit(train[cfg.NUMERIC_FEATURES])
    boot = bootstrap_coefficients(model, train)
    test_proba = model.predict_proba(test[model_columns()])[:, 1]
    balance, propensity = covariate_balance(data)
    return Artifacts(
        data,
        train,
        test,
        audit,
        model,
        meta,
        causal,
        dag,
        pca,
        boot,
        test_proba,
        balance,
        propensity,
    )


def model_columns() -> list[str]:
    """Columnas crudas que espera el pipeline del modelo."""
    return cfg.NUMERIC_FEATURES + cfg.MODEL_BINARY + [cfg.TREATMENT] + cfg.MODEL_CATEGORICAL


# --- Predicción ("What-Is") -----------------------------------------------------------


def bootstrap_coefficients(model, train: pd.DataFrame, n_boot: int = N_BOOT) -> np.ndarray:
    """Coeficientes del clasificador reajustado en remuestras bootstrap del entrenamiento.

    El preprocesamiento queda fijo y solo se reajusta la regresión logística con la misma
    penalización. Cada fila de la matriz es [intercepto, coeficientes].
    """
    pre, clf = model.named_steps["pre"], model.named_steps["modelo"]
    X = pre.transform(train).astype(float).to_numpy()
    y = train[cfg.OUTCOME_BIN].to_numpy()
    rng = np.random.default_rng(cfg.SEED)
    rows = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        fit = clone(clf).fit(X[idx], y[idx])
        rows.append(np.r_[fit.intercept_, fit.coef_[0]])
    return np.array(rows)


def default_profile(train: pd.DataFrame) -> dict:
    """Paciente de referencia: medianas de las numéricas, binarias ausentes y categorías basales."""
    profile = {c: float(train[c].median()) for c in cfg.NUMERIC_FEATURES}
    profile.update({c: 0 for c in cfg.MODEL_BINARY + [cfg.TREATMENT]})
    profile.update({c: cfg.CATEGORY_LEVELS[c][0] for c in cfg.MODEL_CATEGORICAL})
    return profile


def profile_from_row(row: pd.Series) -> dict:
    """Perfil editable a partir de un paciente real del conjunto de datos."""
    profile = {}
    for c in model_columns():
        value = row[c]
        if c in cfg.MODEL_CATEGORICAL:
            profile[c] = str(value)
        elif c in cfg.NUMERIC_FEATURES:
            profile[c] = None if pd.isna(value) else float(value)
        else:
            profile[c] = int(value)
    return profile


def profile_frame(profile: dict) -> pd.DataFrame:
    """Convierte un perfil en una fila con los tipos que espera el preprocesamiento."""
    row = pd.DataFrame([{c: profile[c] for c in model_columns()}])
    for c in cfg.NUMERIC_FEATURES:
        row[c] = pd.to_numeric(row[c], errors="coerce").astype(float)
    for c in cfg.MODEL_CATEGORICAL:
        row[c] = pd.Categorical(row[c], categories=cfg.CATEGORY_LEVELS[c])
    return row


def predict(art: Artifacts, profile: dict) -> dict:
    """Probabilidad de muerte a 30 días con su intervalo de confianza bootstrap del 95 %."""
    row = profile_frame(profile)
    p = float(art.model.predict_proba(row)[:, 1][0])
    x = art.model.named_steps["pre"].transform(row).astype(float).to_numpy()[0]
    draws = 1 / (1 + np.exp(-(art.boot_coefs[:, 0] + art.boot_coefs[:, 1:] @ x)))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"p": p, "ic_inf": float(lo), "ic_sup": float(hi)}


def contributions(art: Artifacts, profile: dict) -> pd.Series:
    """Contribución de cada variable al log-odds de muerte frente a un paciente promedio.

    Con las numéricas estandarizadas, la contribución es β_j·z_j: cuánto aleja el valor del
    paciente su riesgo respecto de uno con valores medios, sin comorbilidades y en las
    categorías de referencia. Las dummies de una misma variable categórica se suman, y el
    indicador de faltante se suma a su variable.
    """
    row = profile_frame(profile)
    pre, clf = art.model.named_steps["pre"], art.model.named_steps["modelo"]
    xt = pre.transform(row).astype(float).iloc[0]
    contrib = pd.Series(clf.coef_[0] * xt.to_numpy(), index=xt.index)
    groups: dict[str, float] = {}
    for name, value in contrib.items():
        base = next((c for c in cfg.MODEL_CATEGORICAL if name.startswith(c + "_")), None)
        if base is None and name.startswith("missingindicator_"):
            base = name.replace("missingindicator_", "")
        key = base or name
        groups[key] = groups.get(key, 0.0) + value
    return pd.Series(groups).sort_values(key=np.abs, ascending=False)


def calibration_bins(art: Artifacts, n_bins: int = 10) -> pd.DataFrame:
    """Diagrama de confiabilidad en test: deciles de riesgo predicho vs. mortalidad observada."""
    df = pd.DataFrame({"p": art.test_proba, "y": art.test[cfg.OUTCOME_BIN].to_numpy()})
    df["bin"] = pd.qcut(df["p"], n_bins, labels=False, duplicates="drop")
    rows = []
    for b, g in df.groupby("bin"):
        lo, hi = wilson_ci(int(g["y"].sum()), len(g))
        rows.append(
            {
                "bin": int(b),
                "desde": g["p"].min(),
                "hasta": g["p"].max(),
                "predicha": g["p"].mean(),
                "observada": g["y"].mean(),
                "ic_inf": lo,
                "ic_sup": hi,
                "n": len(g),
            }
        )
    return pd.DataFrame(rows)


# --- Exploración ---------------------------------------------------------------------


def pca_scores(art: Artifacts, df: pd.DataFrame) -> pd.DataFrame:
    """Proyección de pacientes sobre CP1 y CP2 con el PCA ajustado solo en entrenamiento."""
    scores = art.pca.transform(df[cfg.NUMERIC_FEATURES])
    return pd.DataFrame(scores[:, :2], columns=["CP1", "CP2"], index=df.index)


def pca_loadings(art: Artifacts) -> pd.DataFrame:
    """Cargas de cada variable en CP1 y CP2 y varianza explicada."""
    pca = art.pca.named_steps["pca"]
    out = pd.DataFrame(pca.components_[:2].T, index=cfg.NUMERIC_FEATURES, columns=["CP1", "CP2"])
    out.attrs["varianza"] = pca.explained_variance_ratio_[:2].tolist()
    return out


def smd(a: pd.Series, b: pd.Series) -> float:
    """Diferencia de medias estandarizada (desvío combinado)."""
    a, b = a.dropna().astype(float), b.dropna().astype(float)
    pooled = np.sqrt((a.var() + b.var()) / 2)
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


# --- Decisión causal ("What-If") -----------------------------------------------------


def covariate_balance(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Diferencias de medias estandarizadas antes y después de ponderar por propensión.

    Reproduce el diagnóstico de balance de la Fase 4: puntaje de propensión por regresión
    logística sobre los confusores estandarizados y pesos IPW para el ATE.
    """
    frame, z_cols = causal_frame(df)
    names = build_preprocessor(scale=True, include_treatment=False).fit(df).get_feature_names_out()
    labels = {safe_name(n): n for n in names}
    t = frame[cfg.TREATMENT].to_numpy()
    ps = propensity_scores(frame[z_cols], t)
    w = np.where(t == 1, 1 / ps, 1 / (1 - ps))
    before = standardized_differences(frame[z_cols], t)
    after = standardized_differences(frame[z_cols], t, w)
    balance = pd.DataFrame(
        {
            "variable": [labels.get(c, c) for c in z_cols],
            "antes": before.to_numpy(),
            "despues": after.to_numpy(),
        }
    )
    propensity = pd.DataFrame({"ps": ps, "rhc": t})
    return balance, propensity


def cate_curve(causal: dict, dx: str, age: float) -> pd.DataFrame:
    """Curva de CATE según el APACHE para un diagnóstico, interpolando linealmente en la edad."""
    grid = pd.DataFrame(causal["cate_grilla"])
    grid = grid[grid["dx"] == dx]
    ages = np.sort(grid["edad"].unique())
    age = float(np.clip(age, ages[0], ages[-1]))
    hi_idx = int(np.searchsorted(ages, age))
    lo_age, hi_age = ages[max(hi_idx - 1, 0)], ages[min(hi_idx, len(ages) - 1)]
    w = 0.0 if hi_age == lo_age else (age - lo_age) / (hi_age - lo_age)
    lo = grid[grid["edad"] == lo_age].set_index("aps1").sort_index()
    hi = grid[grid["edad"] == hi_age].set_index("aps1").sort_index()
    cols = ["cate", "ic_inf", "ic_sup"]
    out = (1 - w) * lo[cols] + w * hi[cols]
    return out.reset_index()


def cate_at(causal: dict, dx: str, age: float, aps1: float) -> dict:
    """CATE en el perfil elegido (interpolación lineal en edad y APACHE)."""
    curve = cate_curve(causal, dx, age)
    out = {k: float(np.interp(aps1, curve["aps1"], curve[k])) for k in ["cate", "ic_inf", "ic_sup"]}
    if out["ic_inf"] > 0:
        out["lectura"] = "dañino"
    elif out["ic_sup"] < 0:
        out["lectura"] = "beneficioso"
    else:
        out["lectura"] = "sin efecto demostrable"
    return out


DAG_VIEWS = {
    "Estructura completa": {
        "edges": [],
        "conditioned": [],
        "text": (
            "El grafo resume los supuestos causales del análisis. Los confusores observados Z "
            "influyen en la decisión de cateterizar y en el riesgo de muerte. El efecto del RHC "
            "opera de forma directa y a través de la intensificación terapéutica. Los días de "
            "internación son consecuencia del tratamiento y del resultado. Los nodos con borde "
            "punteado no se observan en los datos."
        ),
    },
    "Caminos causales": {
        "edges": [("T", "Y"), ("T", "M"), ("M", "Y")],
        "conditioned": [],
        "text": (
            "Estos son los caminos que componen el efecto total del RHC sobre la mortalidad: el "
            "efecto directo y el que pasa por la intensificación terapéutica. Ninguno debe "
            "bloquearse, porque son exactamente lo que se quiere estimar."
        ),
    },
    "Caminos de puerta trasera": {
        "edges": [("Z", "T"), ("Z", "Y"), ("U", "T"), ("U", "Y")],
        "conditioned": ["Z"],
        "text": (
            "Los caminos T ← Z → Y generan una asociación entre tratamiento y muerte que no es "
            "causal. Ajustar por Z los bloquea (criterio de puerta trasera). El camino que pasa "
            "por U, el criterio médico y el hospital, no puede bloquearse porque U no se "
            "observa: por eso se calcula el E-value."
        ),
    },
    "Ajustar por la internación": {
        "edges": [("T", "C"), ("Y", "C")],
        "conditioned": ["C"],
        "text": (
            "Los días de internación son un colisionador: el RHC los prolonga y la muerte "
            "temprana los acorta. Si se ajusta por ellos, se abre el camino T → C ← Y y aparece "
            "una asociación espuria (sesgo de Berkson). En los datos, la estimación pasa de "
            "5,4 a 6,9 puntos porcentuales."
        ),
    },
    "Ajustar por el mediador": {
        "edges": [("T", "M"), ("M", "Y")],
        "conditioned": ["M"],
        "text": (
            "Ajustar por la intensificación terapéutica bloquea el camino indirecto: el modelo "
            "estimaría solo el efecto directo y subestimaría el impacto total de la política. "
            "En la simulación del notebook, con un efecto total verdadero de 0,07, el ajuste "
            "por el mediador recupera solo 0,04."
        ),
    },
}


def dag_dot(dag: dict, view: str, palette: dict) -> str:
    """DAG en formato DOT con los arcos y nodos destacados según la vista elegida."""
    spec = DAG_VIEWS[view]
    labels = {
        "Z": "Z  Confusores observados\\ngravedad, diagnóstico,\\ndemografía, pronóstico",
        "U": "U  Criterio médico\\ny hospital",
        "T": "T  Cateterismo\\ncardíaco derecho",
        "M": "M  Intensificación\\nterapéutica",
        "C": "C  Días de\\ninternación",
        "Y": "Y  Muerte\\na 30 días",
    }
    edges = {(a[0], b[0]) for a, b in dag["arcos"]}
    edges = sorted((a, b) for a, b in edges if a != b)
    latent = {n[0] for n in dag["latentes"]}
    lines = [
        "digraph {",
        '  rankdir=LR; bgcolor="transparent"; nodesep=0.3; ranksep=0.6; pad=0.05;',
        '  node [shape=box, style="rounded,filled", fontname="Newsreader", fontsize=17, '
        f'color="{palette["rule"]}", fillcolor="white", fontcolor="{palette["ink"]}", '
        'penwidth=1.3, margin="0.2,0.12"];',
        f'  edge [color="{palette["muted"]}", penwidth=1.5, arrowsize=0.9];',
        "  { rank=same; Z; U; }",
        "  { rank=same; M; C; }",
    ]
    for node, label in labels.items():
        attrs = [f'label="{label}"']
        if node in latent:
            attrs += ['style="rounded,dashed,filled"', f'color="{palette["faint"]}"']
        if node in spec["conditioned"]:
            attrs += [
                f'fillcolor="{palette["fill"]}"',
                f'color="{palette["accent"]}"',
                "penwidth=2.4",
            ]
        lines.append(f"  {node} [{', '.join(attrs)}];")
    for a, b in edges:
        attrs = []
        if a in latent or b in latent:
            attrs.append('style="dashed"')
        if (a, b) in spec["edges"]:
            attrs += [f'color="{palette["highlight"]}"', "penwidth=2.6"]
        lines.append(f"  {a} -> {b} [{', '.join(attrs)}];")
    lines.append("}")
    return "\n".join(lines)
