"""Carga del dataset RHC y construcción del marco analítico tipado."""

from __future__ import annotations

import urllib.request

import numpy as np
import pandas as pd

from rhc.config import (
    ADMISSION_DX,
    COMORBIDITIES,
    DATA_RAW,
    DATA_URL,
    DX_MAP,
    DX_ORDER,
    ID_COLS,
)

YES_NO = {"Yes": 1, "No": 0}


def load_raw(path=DATA_RAW, download_if_missing: bool = True) -> pd.DataFrame:
    """Lee el CSV original. Si no está en disco, lo descarga desde la fuente oficial."""
    if not path.exists():
        if not download_if_missing:
            raise FileNotFoundError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, path)
    return pd.read_csv(path)


def build_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Tipa variables, codifica binarias y deriva la duración de la internación.

    No imputa ni elimina valores: la limpieza vive en ``rhc.cleaning`` para que la
    auditoría forense pueda comparar el dato crudo con el tratado.
    """
    df = raw.drop(columns=[c for c in ID_COLS if c in raw.columns]).copy()

    df["rhc"] = (df.pop("swang1") == "RHC").astype(int)
    for col in ["death", "dth30"]:
        df[col] = df[col].map(YES_NO).astype(int)
    df["dnr1"] = df["dnr1"].map(YES_NO).astype(int)
    for col in ADMISSION_DX:
        df[col] = df[col].map(YES_NO).astype(int)
    df[COMORBIDITIES] = df[COMORBIDITIES].astype(int)

    # Duración de la internación: alta - admisión. El único caso sin fecha de alta
    # (muerte a 394 días con último contacto a 196) es inconsistente: queda como NaN.
    df["los"] = df["dschdte"] - df["sadmdte"]
    df["log_los"] = np.log(df["los"])

    # Diagnóstico primario agrupado: las categorías con menos de 2 % de los casos
    # (cirrosis, cáncer de pulmón y de colon) se unen en "Otros".
    df["dx"] = pd.Categorical(df["cat1"].map(DX_MAP), categories=DX_ORDER)
    df["has_cat2"] = df["cat2"].notna().astype(int)

    income_order = ["Under $11k", "$11-$25k", "$25-$50k", "> $50k"]
    df["income"] = pd.Categorical(df["income"], categories=income_order, ordered=True)
    df["ca"] = pd.Categorical(df["ca"], categories=["No", "Yes", "Metastatic"], ordered=True)
    for col in ["cat1", "cat2", "sex", "race", "ninsclas"]:
        df[col] = df[col].astype("category")
    return df


def load_frame() -> pd.DataFrame:
    """Atajo: del dataset crudo al marco analítico tipado."""
    return build_frame(load_raw())
