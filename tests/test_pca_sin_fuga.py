"""Test de QA: el PCA se ajusta solo con train (sin fuga de datos)."""

import numpy as np

from rhc import config as cfg
from rhc.features import build_pca, load_model_frame, split


def test_pca_se_ajusta_solo_con_train():
    df = load_model_frame()
    train, test = split(df)
    pca = build_pca().fit(train[cfg.NUMERIC_FEATURES])

    # 1. Las medianas de imputación salen de train.
    medianas = pca.named_steps["imputar"].statistics_
    assert np.allclose(medianas, train[cfg.NUMERIC_FEATURES].median().to_numpy())

    # 2. Las medias de la estandarización salen de train y no del dataset completo.
    medias = pca.named_steps["escalar"].mean_
    imputar = pca.named_steps["imputar"]
    assert np.allclose(medias, imputar.transform(train[cfg.NUMERIC_FEATURES]).mean(axis=0))
    assert not np.allclose(medias, imputar.transform(df[cfg.NUMERIC_FEATURES]).mean(axis=0))

    # 3. Transformar test no modifica la rotación aprendida.
    rotacion = pca.named_steps["pca"].components_.copy()
    pca.transform(test[cfg.NUMERIC_FEATURES])
    assert np.array_equal(rotacion, pca.named_steps["pca"].components_)
