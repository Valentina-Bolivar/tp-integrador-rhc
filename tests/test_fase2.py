"""Tests de la Fase 2: partición, matriz de diseño y supuestos estructurales del MCO."""

import numpy as np
import pytest

from rhc import config as cfg
from rhc import regression as reg
from rhc.features import build_preprocessor, load_model_frame, split


@pytest.fixture(scope="module")
def parts():
    df = load_model_frame()
    train, test = split(df)
    return df, train, test


def test_particion_disjunta_y_completa(parts):
    df, train, test = parts
    assert train.index.intersection(test.index).empty
    assert len(train) + len(test) == len(df)
    assert abs(len(test) / len(df) - cfg.TEST_SIZE) < 0.001


def test_particion_reproducible(parts):
    _, train, _ = parts
    train2, _ = split(load_model_frame())
    assert train.index.equals(train2.index), "La semilla debe fijar la partición"


def test_estratificacion_preserva_proporciones(parts):
    _, train, test = parts
    for col in [cfg.OUTCOME_BIN, cfg.TREATMENT]:
        assert abs(train[col].mean() - test[col].mean()) < 0.01


def test_diseno_sin_variables_post_tratamiento(parts):
    _, train, _ = parts
    X = build_preprocessor().fit_transform(train)
    assert set(X.columns).isdisjoint(cfg.POST_TREATMENT)
    assert X.isna().sum().sum() == 0, "La imputación debe completar todos los faltantes"


def test_imputacion_usa_medianas_de_train(parts):
    _, train, _ = parts
    pre = build_preprocessor().fit(train)
    imputer = pre.named_transformers_["num_ind"].named_steps["imputar"]
    expected = train[cfg.INDICATOR_VARS].median().to_numpy()
    assert np.allclose(imputer.statistics_, expected)


def test_test_conserva_columnas_de_train(parts):
    _, train, test = parts
    pre = build_preprocessor().fit(train)
    assert list(pre.transform(test).columns) == list(pre.transform(train).columns)


def test_sin_multicolinealidad_severa(parts):
    _, train, _ = parts
    X = build_preprocessor().fit_transform(train.dropna(subset=["log_los"]))
    assert reg.vif(X).max() < 10, "Ningún regresor del diseño final debe superar VIF = 10"
