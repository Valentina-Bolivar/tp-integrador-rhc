"""Contrastes de hipótesis con intervalos de confianza y tamaños de efecto (Fase 1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.oneway import anova_oneway
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize


def fmt_p(p: float) -> str:
    """Formatea un p-valor exacto en notación científica legible (nunca '0.000')."""
    if p >= 0.001:
        return f"{p:.4f}"
    if p < 1e-16:  # por debajo de la precisión de máquina
        return "< 10⁻¹⁶"
    mant, exp = f"{p:.2e}".split("e")
    sup = str(int(exp)).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{mant} × 10{sup}"


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Intervalo de Wilson para una proporción."""
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    center = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return center - half, center + half


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """d de Cohen con desvío combinado (pooled)."""
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / sp


def welch_test(a: np.ndarray, b: np.ndarray, alpha: float = 0.05) -> dict:
    """Test t de Welch (varianzas desiguales) con IC para μa − μb y d de Cohen."""
    res = stats.ttest_ind(a, b, equal_var=False)
    ci = res.confidence_interval(confidence_level=1 - alpha)
    return {
        "estadístico": "t de Welch",
        "valor": res.statistic,
        "gl": res.df,
        "p_valor": res.pvalue,
        "diferencia": a.mean() - b.mean(),
        "IC_inf": ci.low,
        "IC_sup": ci.high,
        "efecto": cohen_d(a, b),
        "medida_efecto": "d de Cohen",
    }


def mann_whitney(a: np.ndarray, b: np.ndarray, alpha: float = 0.05, seed: int = 42) -> dict:
    """U de Mann-Whitney, biserial por rangos e IC bootstrap de la diferencia de medianas."""
    res = stats.mannwhitneyu(a, b, alternative="two-sided")
    rbc = 1 - 2 * res.statistic / (len(a) * len(b))
    rng = np.random.default_rng(seed)
    boots = [
        np.median(rng.choice(a, len(a))) - np.median(rng.choice(b, len(b))) for _ in range(2000)
    ]
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "estadístico": "U de Mann-Whitney",
        "valor": res.statistic,
        "gl": np.nan,
        "p_valor": res.pvalue,
        "diferencia": np.median(a) - np.median(b),
        "IC_inf": lo,
        "IC_sup": hi,
        "efecto": -rbc,
        "medida_efecto": "biserial por rangos",
    }


def two_proportions(ya: np.ndarray, yb: np.ndarray, alpha: float = 0.05) -> dict:
    """Test z de dos proporciones (≡ χ² 2×2), IC de Wald para pa − pb y h de Cohen."""
    pa, pb, na, nb = ya.mean(), yb.mean(), len(ya), len(yb)
    p = (ya.sum() + yb.sum()) / (na + nb)
    z = (pa - pb) / np.sqrt(p * (1 - p) * (1 / na + 1 / nb))
    se = np.sqrt(pa * (1 - pa) / na + pb * (1 - pb) / nb)
    q = stats.norm.ppf(1 - alpha / 2)
    return {
        "estadístico": "z de dos proporciones",
        "valor": z,
        "gl": np.nan,
        "p_valor": 2 * stats.norm.sf(abs(z)),
        "diferencia": pa - pb,
        "IC_inf": pa - pb - q * se,
        "IC_sup": pa - pb + q * se,
        "efecto": proportion_effectsize(pa, pb),
        "medida_efecto": "h de Cohen",
    }


def power_two_proportions(h: float, n1: int, n2: int, alpha: float = 0.05) -> float:
    """Potencia (1 − β) del test z de dos proporciones para un efecto h."""
    return NormalIndPower().power(effect_size=h, nobs1=n1, ratio=n2 / n1, alpha=alpha)


def min_detectable_h(n1: int, n2: int, alpha: float = 0.05, power: float = 0.80) -> float:
    """Menor efecto h detectable con la potencia indicada (análisis de sensibilidad)."""
    return NormalIndPower().solve_power(nobs1=n1, ratio=n2 / n1, alpha=alpha, power=power)


def normality_profile(groups: dict[str, np.ndarray]) -> pd.DataFrame:
    """Asimetría, curtosis en exceso y Shapiro-Wilk por grupo (este último solo informativo)."""
    rows = []
    for name, x in groups.items():
        sw = stats.shapiro(x if len(x) <= 5000 else np.random.default_rng(0).choice(x, 5000))
        rows.append(
            {
                "grupo": name,
                "n": len(x),
                "asimetría": stats.skew(x),
                "curtosis_exceso": stats.kurtosis(x),
                "Shapiro-Wilk W": sw.statistic,
                "p_SW": sw.pvalue,
            }
        )
    return pd.DataFrame(rows)


def anova_battery(df: pd.DataFrame, y: str, factor: str) -> dict:
    """ANOVA clásico, Levene (Brown-Forsythe), Welch, Kruskal-Wallis y tamaños de efecto."""
    groups = [g[y].to_numpy() for _, g in df.groupby(factor, observed=True)]
    n, k = len(df), len(groups)
    f_cl = stats.f_oneway(*groups)
    lev = stats.levene(*groups, center="median")
    welch = anova_oneway(df[y], df[factor], use_var="unequal")
    kw = stats.kruskal(*groups)

    grand = df[y].mean()
    ss_b = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_t = ((df[y] - grand) ** 2).sum()
    ms_w = (ss_t - ss_b) / (n - k)
    return {
        "F_clásico": f_cl.statistic,
        "p_F": f_cl.pvalue,
        "gl": (k - 1, n - k),
        "Levene_W": lev.statistic,
        "p_Levene": lev.pvalue,
        "F_Welch": welch.statistic,
        "gl_Welch": (welch.df_num, welch.df_denom),
        "p_Welch": welch.pvalue,
        "H_Kruskal": kw.statistic,
        "p_Kruskal": kw.pvalue,
        "eta2": ss_b / ss_t,
        "omega2": (ss_b - (k - 1) * ms_w) / (ss_t + ms_w),
        "epsilon2_KW": kw.statistic / ((n**2 - 1) / (n + 1)),
    }


def posthoc(df: pd.DataFrame, y: str, factor: str, alpha: float = 0.05) -> pd.DataFrame:
    """Comparaciones múltiples: Tukey HSD frente a t de Welch con corrección de Bonferroni."""
    tk = pairwise_tukeyhsd(df[y], df[factor].astype(str), alpha=alpha)
    tk_df = pd.DataFrame(tk.summary().data[1:], columns=tk.summary().data[0])
    tk_df = tk_df.rename(columns={"p-adj": "p_Tukey", "meandiff": "dif_medias"})

    pvals = []
    for _, r in tk_df.iterrows():
        a = df.loc[df[factor].astype(str) == r["group1"], y]
        b = df.loc[df[factor].astype(str) == r["group2"], y]
        pvals.append(stats.ttest_ind(a, b, equal_var=False).pvalue)
    tk_df["p_Bonferroni"] = multipletests(pvals, alpha=alpha, method="bonferroni")[1]
    tk_df["rechaza_Tukey"] = tk_df["p_Tukey"] < alpha
    tk_df["rechaza_Bonferroni"] = tk_df["p_Bonferroni"] < alpha
    return tk_df[
        ["group1", "group2", "dif_medias", "p_Tukey", "p_Bonferroni",
         "rechaza_Tukey", "rechaza_Bonferroni"]
    ]
