"""Tests de la Fase 3: clasificador Naive Bayes mixto, métricas y modelo serializado."""

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.naive_bayes import GaussianNB

from rhc import config as cfg
from rhc import models as mdl
from rhc.features import load_model_frame, split


@pytest.fixture(scope="module")
def parts():
    train, test = split(load_model_frame())
    return train, test


def test_naive_bayes_probabilidades_validas(parts):
    train, test = parts
    nb = mdl.build_naive_bayes().fit(train, train[cfg.OUTCOME_BIN])
    proba = nb.predict_proba(test)
    assert np.allclose(proba.sum(axis=1), 1)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_naive_bayes_mixto_cuenta_el_prior_una_sola_vez():
    """Con clases balanceadas, una binaria constante no aporta información (ni siquiera por
    el suavizado de Laplace), y el modelo mixto debe coincidir con GaussianNB."""
    rng = np.random.default_rng(0)
    y = pd.Series([0] * 150 + [1] * 150)
    values = rng.normal(size=(300, 2)) + y.to_numpy()[:, None]
    X = pd.DataFrame(values, columns=cfg.NUMERIC_FEATURES[:2])
    X["dummy"] = 1.0
    mixed = mdl.MixedNaiveBayes().fit(X, y)
    gauss = GaussianNB().fit(X.iloc[:, :2].to_numpy(), y)
    expected = gauss.predict_proba(X.iloc[:, :2].to_numpy())
    assert np.allclose(mixed.predict_proba(X), expected, atol=1e-6)


def test_metricas_de_un_clasificador_perfecto():
    y = np.array([0, 0, 1, 1])
    m = mdl.classification_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert m["AUC-ROC"] == 1 and m["F1"] == 1 and m["Brier"] < 0.05


def test_bootstrap_de_modelos_identicos_es_nulo():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    p = rng.random(500)
    r = mdl.bootstrap_auc_difference(y, p, p, n_boot=200)
    assert r["diferencia"] == 0 and r["IC_inf"] == 0 and r["IC_sup"] == 0


def test_modelo_serializado_carga_y_predice(parts):
    _, test = parts
    path = cfg.MODELS_DIR / "modelo_logistico.joblib"
    if not path.exists():
        pytest.skip("El modelo se genera al ejecutar el notebook")
    model = joblib.load(path)
    proba = model.predict_proba(test)[:, 1]
    meta = json.loads((cfg.MODELS_DIR / "modelo_logistico.json").read_text())
    assert len(proba) == len(test)
    auc = mdl.roc_auc_score(test[cfg.OUTCOME_BIN], proba)
    assert abs(meta["metricas_test"]["AUC-ROC"] - auc) < 1e-3
    assert 0 < meta["umbral_youden"] < 1
