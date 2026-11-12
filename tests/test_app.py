"""Tests de la web app: capa de servicio y ejecución completa con el arnés de Streamlit."""

from __future__ import annotations

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from rhc import config as cfg
from rhc import serving as sv


@pytest.fixture(scope="module")
def art():
    return sv.load_artifacts()


def test_prediccion_dentro_del_intervalo(art):
    profile = sv.default_profile(art.train)
    pred = sv.predict(art, profile)
    assert 0 < pred["ic_inf"] <= pred["p"] <= pred["ic_sup"] < 1


def test_contribuciones_reconstruyen_la_probabilidad(art):
    """Intercepto + suma de contribuciones = logit de la probabilidad predicha."""
    row = art.test.iloc[0]
    profile = sv.profile_from_row(row)
    p = sv.predict(art, profile)["p"]
    intercept = art.model.named_steps["modelo"].intercept_[0]
    logit = intercept + sv.contributions(art, profile).sum()
    assert np.isclose(1 / (1 + np.exp(-logit)), p, atol=1e-9)


def test_faltantes_se_imputan(art):
    profile = sv.default_profile(art.train) | {"wtkilo1": None, "hrt1": None}
    pred = sv.predict(art, profile)
    assert np.isfinite(pred["p"])


def test_calibracion_cubre_todo_el_test(art):
    cal = sv.calibration_bins(art)
    assert cal["n"].sum() == len(art.test)
    assert (cal["ic_inf"] <= cal["observada"]).all() and (cal["observada"] <= cal["ic_sup"]).all()


def test_cate_coincide_con_la_grilla(art):
    grid = art.causal["cate_grilla"]
    point = next(g for g in grid if g["edad"] == 70 and g["aps1"] == 80)
    out = sv.cate_at(art.causal, point["dx"], 70, 80)
    assert np.isclose(out["cate"], point["cate"])
    expected = "dañino" if point["ic_inf"] > 0 else None
    if expected:
        assert out["lectura"] == expected


def test_ponderacion_balancea_los_confusores(art):
    """Tras ponderar por propensión ninguna covariable supera |SMD| = 0,1 (Fase 4)."""
    assert (art.balance["antes"].abs() > 0.1).sum() > 0
    assert (art.balance["despues"].abs() <= 0.1).all()


@pytest.mark.parametrize("view", list(sv.DAG_VIEWS))
def test_dag_tiene_todos_los_nodos(art, view):
    keys = ["rule", "muted", "ink", "fill", "accent", "highlight", "faint"]
    palette = dict.fromkeys(keys, "#000000")
    dot = sv.dag_dot(art.dag, view, palette)
    for node in ["Z", "U", "T", "M", "C", "Y"]:
        assert f"  {node} [" in dot
    assert dot.count("->") == 9


def test_app_corre_sin_errores():
    at = AppTest.from_file(str(cfg.ROOT / "app" / "app.py"), default_timeout=300)
    at.run()
    assert not at.exception
    assert len(at.tabs) == 4
    next(b for b in at.button if b.label == "Paciente real al azar").click().run()
    assert not at.exception
    at.selectbox(key="in_dx").set_value("Coma").run()
    assert at.session_state["c_dx"] == "Coma"
    at.button_group(key="f_dag").set_value("Ajustar por la internación").run()
    at.slider(key="c_aps").set_value(120).run()
    assert not at.exception


def test_verificaciones_causales_coinciden_con_el_notebook(art):
    """El ajuste por Z recalculado en la app reproduce el ATE por regresión guardado en models/."""
    chk = art.checks
    ate_lr = art.causal["ate"]["DoWhy: regresión con ajuste backdoor"]["valor"]
    assert abs(chk["ajuste_z"] - ate_lr) < 1e-4
    assert chk["ajuste_z_colisionador"] > chk["ajuste_z"]
    assert 0 < chk["recorte_proporcion"] < 0.10
    assert chk["sim_mediador"] < chk["sim_total"]
