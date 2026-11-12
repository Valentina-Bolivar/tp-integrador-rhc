"""Tests de la Fase 1: reglas de auditoría y funciones de inferencia."""

import numpy as np
import pytest

from rhc import inference as inf
from rhc.cleaning import LEGITIMATE_ZEROS, SENTINEL_RULES, apply_sentinels
from rhc.data import load_frame


@pytest.fixture(scope="module")
def frames():
    df = load_frame()
    clean, audit = apply_sentinels(df)
    return df, clean, audit


def test_centinelas_eliminados(frames):
    _, clean, _ = frames
    for col in ["wtkilo1", "hrt1", "meanbp1", "resp1", "wblc1"]:
        assert (clean[col] == 0).sum() == 0, f"Quedan ceros imposibles en {col}"
    assert (clean["alb1"] > 10).sum() == 0


def test_auditoria_no_pierde_filas(frames):
    df, clean, audit = frames
    assert len(clean) == len(df), "La limpieza no debe eliminar pacientes"
    assert audit["casos"].sum() == 896


def test_ceros_legitimos_se_conservan(frames):
    df, clean, _ = frames
    for col in LEGITIMATE_ZEROS:
        assert (clean[col] == 0).sum() == (df[col] == 0).sum()


def test_reglas_no_tocan_ceros_legitimos():
    assert {r.column for r in SENTINEL_RULES}.isdisjoint(LEGITIMATE_ZEROS)


def test_cat2_sin_faltantes(frames):
    _, clean, _ = frames
    assert clean["cat2"].isna().sum() == 0


def test_welch_recupera_diferencia_conocida():
    rng = np.random.default_rng(0)
    a, b = rng.normal(10, 1, 5000), rng.normal(9, 3, 5000)
    r = inf.welch_test(a, b)
    assert r["IC_inf"] < 1 < r["IC_sup"] and r["p_valor"] < 0.001


def test_dos_proporciones_cobertura_del_ic():
    """Bajo H0, el IC del 95 % debe contener la diferencia real (0) en ~95 % de las réplicas."""
    rng = np.random.default_rng(1)
    hits = 0
    for _ in range(1000):
        r = inf.two_proportions(rng.binomial(1, 0.3, 2000), rng.binomial(1, 0.3, 2000))
        hits += r["IC_inf"] < 0 < r["IC_sup"]
    assert 0.93 <= hits / 1000 <= 0.97


def test_wilson_contiene_proporcion():
    lo, hi = inf.wilson_ci(30, 100)
    assert lo < 0.30 < hi and 0 <= lo and hi <= 1


def test_fmt_p_nunca_muestra_cero():
    assert inf.fmt_p(0.0) == "< 10⁻¹⁶"
    assert "10" in inf.fmt_p(1e-20) and inf.fmt_p(0.0472) == "0.0472"
