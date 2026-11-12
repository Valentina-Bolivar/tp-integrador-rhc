"""Contrato de datos: integridad de dimensiones, tipos y ausencia de fuga de información."""

import pytest

from rhc.config import (
    BINARY_FEATURES,
    CATEGORICAL,
    NUMERIC_FEATURES,
    OUTCOME_BIN,
    POST_TREATMENT,
    TREATMENT,
)
from rhc.data import load_frame


@pytest.fixture(scope="module")
def df():
    return load_frame()


def test_dimensiones(df):
    assert len(df) == 5735, "El dataset RHC debe tener 5.735 pacientes"


def test_tratamiento_y_resultado_binarios(df):
    assert set(df[TREATMENT].unique()) == {0, 1}
    assert set(df[OUTCOME_BIN].unique()) == {0, 1}
    assert df[TREATMENT].sum() == 2184, "Deben existir 2.184 pacientes tratados"


def test_duracion_internacion_positiva(df):
    assert (df["los"].dropna() > 0).all()


def test_sin_identificadores(df):
    assert "ptid" not in df.columns and "Unnamed: 0" not in df.columns


def test_predictoras_no_incluyen_variables_post_tratamiento():
    features = set(NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL)
    assert features.isdisjoint(POST_TREATMENT), "Fuga de datos: hay variables post-tratamiento"
    assert TREATMENT not in features


def test_cumple_requisitos_del_enunciado(df):
    n_explicativas = len(NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL)
    assert len(df) >= 1500 and n_explicativas >= 10
