"""Tests de la Fase 4: estructura del DAG, estimadores causales y resultados guardados."""

import json

import networkx as nx
import numpy as np
import pytest

from rhc import causal as cz
from rhc import config as cfg


def test_dag_conceptual_es_aciclico():
    assert nx.is_directed_acyclic_graph(nx.DiGraph(cz.DAG_EDGES))


def test_estadia_es_colisionador_y_no_confusor():
    g = cz.dowhy_graph(["z1", "z2"])
    assert nx.is_directed_acyclic_graph(g)
    assert set(g.predecessors(cz.COLLIDER)) == {cfg.TREATMENT, cfg.OUTCOME_BIN}
    assert cz.COLLIDER not in nx.ancestors(g, cfg.TREATMENT)


def test_e_value_valores_conocidos():
    assert cz.e_value(1.0) == pytest.approx(1.0)
    assert cz.e_value(2.0) == pytest.approx(2 + np.sqrt(2))
    assert cz.e_value(0.5) == pytest.approx(cz.e_value(2.0)), "Un RR protector se invierte"


def test_hajek_recupera_el_ate_en_un_experimento_simulado():
    rng = np.random.default_rng(0)
    n = 50_000
    z = rng.normal(size=n)
    ps = 1 / (1 + np.exp(-z))
    t = rng.binomial(1, ps)
    y = rng.binomial(1, np.clip(0.3 + 0.05 * t + 0.1 * (z > 0), 0, 1))
    assert cz.hajek_ipw(y, t, ps)["ATE"] == pytest.approx(0.05, abs=0.015)


def test_simulacion_de_las_preguntas_trampa():
    sim = cz.simulate_trap_questions(n=100_000)
    est = sim["efecto estimado de T"]
    assert est["Z (criterio backdoor)"] == pytest.approx(0.07, abs=0.005)
    # Ajustar por el mediador deja solo el efecto directo (0,04).
    assert est["Z + mediador M"] == pytest.approx(0.04, abs=0.005)
    assert abs(est["Z + colisionador C"] - 0.07) > 0.03, "Ajustar por C debe sesgar la estimación"


def test_resultados_causales_guardados():
    path = cfg.MODELS_DIR / "resultados_causales.json"
    if not path.exists():
        pytest.skip("Los resultados se generan al ejecutar el notebook")
    res = json.loads(path.read_text())
    assert all(r["supera"] for r in res["refutaciones"].values())
    assert abs(res["refutaciones"]["Tratamiento placebo"]["ate_refutado"]) < 0.01
    for est in res["ate"].values():
        assert est["ic_inf"] < est["valor"] < est["ic_sup"]
    assert len(res["cate_grilla"]) > 0 and res["e_value"]["puntual"] >= 1
