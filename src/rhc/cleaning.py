"""Auditoría forense: reglas de valores centinela y diagnóstico de faltantes.

Las reglas son declarativas para que cada decisión quede documentada, testeada y
citada en la Bitácora. Ninguna función imputa: la imputación se ajusta solo sobre
el conjunto de entrenamiento dentro de los pipelines de modelado (sin fuga).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass(frozen=True)
class SentinelRule:
    column: str
    condition: str  # expresión evaluada con pandas.eval sobre la columna `x`
    rationale: str


# Valores físicamente imposibles en un paciente internado vivo al ingreso.
SENTINEL_RULES = [
    SentinelRule("wtkilo1", "x == 0", "Peso de 0 kg: dato no registrado codificado como cero"),
    SentinelRule("hrt1", "x == 0", "Frecuencia cardíaca 0 lpm: no registrada (o paro al medir)"),
    SentinelRule("meanbp1", "x == 0", "Presión arterial media 0 mmHg: no registrada"),
    SentinelRule("resp1", "x == 0", "Frecuencia respiratoria 0 rpm: no registrada"),
    SentinelRule("wblc1", "x == 0", "Leucocitos 0: incompatible con la vida"),
    SentinelRule("alb1", "x > 10", "Albúmina > 10 g/dL: error de tipeo (rango fisiológico 2 a 6)"),
]

# Ceros legítimos: se documentan para no confundirlos con centinelas.
LEGITIMATE_ZEROS = {
    "scoma1": "Escala de coma SUPPORT: 0 = paciente sin coma",
    "urin1": "Diuresis 0 = anuria, cuadro clínico real",
    "edu": "0 años de educación formal: valor posible",
}

# Faltantes estructurales: la ausencia es una categoría válida, no un dato perdido.
STRUCTURAL_MISSING = {"cat2": "Sin diagnóstico secundario"}

# Variables con faltantes masivos que se excluyen del modelado principal.
EXCLUDED_HIGH_MISSING = ["adld3p", "urin1"]


def apply_sentinels(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reemplaza centinelas por NaN y devuelve el marco limpio y el informe de auditoría."""
    out = df.copy()
    rows = []
    for rule in SENTINEL_RULES:
        x = out[rule.column]
        mask = pd.eval(rule.condition, local_dict={"x": x})
        before = x.describe()
        out.loc[mask, rule.column] = np.nan
        after = out[rule.column].describe()
        rows.append(
            {
                "variable": rule.column,
                "regla": rule.condition.replace("x", rule.column),
                "casos": int(mask.sum()),
                "media_cruda": before["mean"],
                "media_limpia": after["mean"],
                "sesgo_media_%": 100 * (before["mean"] - after["mean"]) / after["mean"],
                "desvío_crudo": before["std"],
                "desvío_limpio": after["std"],
                "justificación": rule.rationale,
            }
        )
    for col, label in STRUCTURAL_MISSING.items():
        out[col] = out[col].cat.add_categories(label).fillna(label)
    return out, pd.DataFrame(rows)


def missingness_table(raw: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    """Porcentaje de faltantes explícitos (NaN en el CSV) y ocultos (centinelas)."""
    cols = sorted(set(raw.columns) & set(clean.columns))
    explicit = raw[cols].isna().mean()
    total = clean[cols].isna().mean()
    tab = pd.DataFrame({"explícito_%": 100 * explicit, "total_%": 100 * total})
    tab["oculto_%"] = tab["total_%"] - tab["explícito_%"]
    return tab[tab["total_%"] > 0].sort_values("total_%", ascending=False)


def missingness_mechanism(
    df: pd.DataFrame,
    column: str,
    predictors: list[str],
    treatment: str,
    outcome: str,
) -> dict:
    """Diagnostica el mecanismo de pérdida de una variable.

    1. Bajo MCAR, el indicador de faltante R es independiente de todo lo observado.
       Se ajusta un logit R ~ Z y se contrasta H0: todos los coeficientes = 0 con
       el test de razón de verosimilitud (LR ~ chi2 con k g.l.).
    2. Si R se asocia con el resultado Y aun condicionando en Z, la pérdida depende
       de información no capturada por Z: indicio de MNAR (no demostrable, porque el
       valor faltante es inobservable por definición).
    """
    r = df[column].isna().astype(int)
    z = df[predictors].copy()
    z = z.fillna(z.median(numeric_only=True))
    x = sm.add_constant(pd.concat([z, df[[treatment]]], axis=1).astype(float))
    fit = sm.Logit(r, x).fit(disp=False)

    x_y = sm.add_constant(pd.concat([z, df[[treatment, outcome]]], axis=1).astype(float))
    fit_y = sm.Logit(r, x_y).fit(disp=False)

    return {
        "variable": column,
        "faltante_%": 100 * r.mean(),
        "LR_chi2 (R~Z,T)": fit.llr,
        "gl": int(fit.df_model),
        "p_MCAR": fit.llr_pvalue,
        "OR_tratamiento": float(np.exp(fit.params[treatment])),
        "OR_resultado | Z,T": float(np.exp(fit_y.params[outcome])),
        "p_resultado | Z,T": float(fit_y.pvalues[outcome]),
    }
