"""Modelo lineal clásico (MCO) y auditoría de los supuestos de Gauss-Markov (Fase 2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import OLSInfluence, variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera


def fit_ols(X: pd.DataFrame, y: pd.Series):
    """Ajusta y = Xβ + ε por MCO con constante."""
    return sm.OLS(y, sm.add_constant(X.astype(float))).fit()


def coefficient_table(model, cov_type: str = "HC3", alpha: float = 0.05) -> pd.DataFrame:
    """Coeficientes con errores estándar clásicos y robustos (HC3) y su efecto porcentual.

    Con respuesta log(y), un cambio unitario en x_j multiplica y por exp(β_j):
    el efecto porcentual es 100·(exp(β_j) − 1).
    """
    robust = model.get_robustcov_results(cov_type)
    ci = robust.conf_int(alpha)
    tab = pd.DataFrame(
        {
            "beta": model.params,
            "ee_MCO": model.bse,
            f"ee_{cov_type}": robust.bse,
            "t": robust.tvalues,
            "p_valor": robust.pvalues,
            "IC_inf": ci[:, 0],
            "IC_sup": ci[:, 1],
        },
        index=model.params.index,
    )
    tab["efecto_%"] = 100 * (np.exp(tab["beta"]) - 1)
    tab["efecto_%_inf"] = 100 * (np.exp(tab["IC_inf"]) - 1)
    tab["efecto_%_sup"] = 100 * (np.exp(tab["IC_sup"]) - 1)
    return tab.drop(index="const")


def gauss_markov_tests(model, order: np.ndarray | None = None) -> pd.DataFrame:
    """Breusch-Pagan (homocedasticidad), Jarque-Bera (normalidad), Durbin-Watson (independencia).

    ``order`` reordena los residuos (por ejemplo, por fecha de admisión) antes de calcular
    Durbin-Watson, que solo tiene sentido sobre una secuencia temporal o espacial.
    """
    resid = model.resid.to_numpy()
    lm, lm_p, f, f_p = het_breuschpagan(resid, model.model.exog)
    jb, jb_p, skew, kurt = jarque_bera(resid)
    dw = durbin_watson(resid[order] if order is not None else resid)
    k = int(model.df_model)
    return pd.DataFrame(
        {
            "estadístico": [lm, jb, dw],
            "g.l.": [k, 2, np.nan],
            "p_valor": [lm_p, jb_p, np.nan],
            "detalle": [
                f"F = {f:.2f} (p = {f_p:.2e})",
                f"asimetría = {skew:.3f}; curtosis = {kurt:.3f}",
                "valores cercanos a 2 indican ausencia de autocorrelación",
            ],
        },
        index=["Breusch-Pagan (LM)", "Jarque-Bera", "Durbin-Watson"],
    )


def influence_table(model) -> pd.DataFrame:
    """Residuos estudentizados externamente, leverage h_ii y distancia de Cook por observación."""
    inf = OLSInfluence(model)
    return pd.DataFrame(
        {
            "r_ext": inf.resid_studentized_external,
            "leverage": inf.hat_matrix_diag,
            "cook": inf.cooks_distance[0],
            "ajustado": model.fittedvalues,
        },
        index=model.resid.index,
    )


def influence_thresholds(n: int, p: int) -> dict:
    """Umbrales del enunciado: |r| > 3, h > 2(p+1)/n, D > 4/n y D > 0,5."""
    return {"r": 3.0, "leverage": 2 * (p + 1) / n, "cook_4n": 4 / n, "cook_abs": 0.5}


def vif(X: pd.DataFrame) -> pd.Series:
    """VIF_j = 1 / (1 − R²_j), calculado sobre la matriz de diseño con constante."""
    Xc = sm.add_constant(X.astype(float)).to_numpy()
    values = [variance_inflation_factor(Xc, i) for i in range(1, Xc.shape[1])]
    return pd.Series(values, index=X.columns, name="VIF").sort_values(ascending=False)


def fit_huber(X: pd.DataFrame, y: pd.Series):
    """Regresión robusta M-estimador de Huber (pondera a la baja los residuos grandes)."""
    return sm.RLM(y, sm.add_constant(X.astype(float)), M=sm.robust.norms.HuberT()).fit()


def r2_out_of_sample(model, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    """R² en test usando la media de test como referencia."""
    pred = model.predict(sm.add_constant(X_test.astype(float), has_constant="add"))
    return 1 - ((y_test - pred) ** 2).sum() / ((y_test - y_test.mean()) ** 2).sum()
