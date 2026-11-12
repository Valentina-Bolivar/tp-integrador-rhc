"""Partición train/test y preprocesamiento sin fuga de datos (Fases 2 a 4).

Contrato: todo parámetro aprendido de los datos (medianas de imputación, medias y
desvíos del escalado, rotación del PCA, categorías de las dummies) se estima
exclusivamente sobre el conjunto de entrenamiento y luego se aplica a test.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from rhc.cleaning import apply_sentinels
from rhc.config import (
    CATEGORY_LEVELS,
    INDICATOR_VARS,
    MODEL_BINARY,
    MODEL_CATEGORICAL,
    NUMERIC_FEATURES,
    OUTCOME_BIN,
    SEED,
    TEST_SIZE,
    TREATMENT,
)
from rhc.data import load_frame


def load_model_frame() -> pd.DataFrame:
    """Marco analítico con los centinelas ya recodificados como faltantes."""
    clean, _ = apply_sentinels(load_frame())
    return clean


def split(df: pd.DataFrame, test_size: float = TEST_SIZE, seed: int = SEED):
    """Partición 80/20 estratificada por la combinación resultado × tratamiento.

    Estratificar por ambas variables preserva en train y en test la mortalidad y la
    proporción de tratados, lo que requieren las Fases 3 (clasificación) y 4 (causal).
    """
    strata = df[OUTCOME_BIN].astype(str) + "_" + df[TREATMENT].astype(str)
    return train_test_split(df, test_size=test_size, random_state=seed, stratify=strata)


def build_preprocessor(
    scale: bool = False,
    drop_first: bool = True,
    binary: list[str] | None = None,
    include_treatment: bool = True,
) -> ColumnTransformer:
    """Imputación por mediana, indicadores de faltante, escalado opcional y dummies.

    Los indicadores de faltante se generan solo para las variables de ``INDICATOR_VARS``
    (pérdida analizada en la Fase 1). Un indicador con muy pocos casos (por ejemplo, la
    única albúmina recodificada) tendría apalancamiento igual a 1 y degeneraría el diseño.

    ``drop_first=True`` elimina una categoría de referencia por variable para evitar la
    trampa de las variables ficticias en el MCO (colinealidad perfecta con la constante).
    ``include_treatment=False`` construye solo las covariables (el conjunto de ajuste Z de
    la Fase 4), sin el tratamiento.
    """
    other_num = [c for c in NUMERIC_FEATURES if c not in INDICATOR_VARS]
    with_ind = [("imputar", SimpleImputer(strategy="median", add_indicator=True))]
    without_ind = [("imputar", SimpleImputer(strategy="median"))]
    if scale:
        with_ind.append(("escalar", StandardScaler()))
        without_ind.append(("escalar", StandardScaler()))
    categorical = OneHotEncoder(
        categories=[CATEGORY_LEVELS[c] for c in MODEL_CATEGORICAL],
        drop="first" if drop_first else None,
        handle_unknown="ignore" if not drop_first else "error",
        sparse_output=False,
    )
    binary_cols = (binary or MODEL_BINARY) + ([TREATMENT] if include_treatment else [])
    pre = ColumnTransformer(
        [
            ("num", Pipeline(without_ind), other_num),
            ("num_ind", Pipeline(with_ind), INDICATOR_VARS),
            ("bin", "passthrough", binary_cols),
            ("cat", categorical, MODEL_CATEGORICAL),
        ],
        verbose_feature_names_out=False,
    )
    return pre.set_output(transform="pandas")


def build_pca(n_components: int | None = None) -> Pipeline:
    """Imputación, estandarización y PCA sobre las variables numéricas continuas."""
    return Pipeline(
        [
            ("imputar", SimpleImputer(strategy="median")),
            ("escalar", StandardScaler()),
            ("pca", PCA(n_components=n_components, random_state=SEED)),
        ]
    )
