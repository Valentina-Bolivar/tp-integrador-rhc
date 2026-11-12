"""Torneo predictivo de la Fase 3: cuatro familias de clasificadores y su evaluación.

Cada modelo es un Pipeline (preprocesamiento + clasificador). En la validación cruzada,
la imputación y el escalado se reajustan dentro de cada pliegue: el pliegue de validación
nunca aporta información al entrenamiento.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from rhc.config import CV_FOLDS, NUMERIC_FEATURES, SEED
from rhc.features import build_preprocessor


def cv_splitter() -> StratifiedKFold:
    """Validación cruzada estratificada de 5 pliegues, con semilla fija."""
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)


class MixedNaiveBayes(ClassifierMixin, BaseEstimator):
    """Naive Bayes para variables mixtas: gaussiana para continuas, Bernoulli para binarias.

    Bajo el supuesto de independencia condicional,
    log P(y | x) = log P(y) + Σ log P(x_continuas | y) + Σ log P(x_binarias | y) + cte.
    Cada componente aporta su log-verosimilitud conjunta, que incluye log P(y); el prior se
    resta una vez para no contarlo dos veces.
    """

    def __init__(self, var_smoothing: float = 1e-9, alpha: float = 1.0):
        self.var_smoothing = var_smoothing
        self.alpha = alpha

    def _split(self, X: pd.DataFrame):
        num = [c for c in X.columns if c in NUMERIC_FEATURES]
        binary = [c for c in X.columns if c not in NUMERIC_FEATURES]
        return X[num].to_numpy(float), X[binary].to_numpy(float)

    def fit(self, X, y):
        xn, xb = self._split(X)
        self.gnb_ = GaussianNB(var_smoothing=self.var_smoothing).fit(xn, y)
        self.bnb_ = BernoulliNB(alpha=self.alpha, binarize=0.5).fit(xb, y)
        self.classes_ = self.gnb_.classes_
        self.log_prior_ = np.log(self.gnb_.class_prior_)
        return self

    def predict_joint_log_proba(self, X):
        xn, xb = self._split(X)
        return (
            self.gnb_.predict_joint_log_proba(xn)
            + self.bnb_.predict_joint_log_proba(xb)
            - self.log_prior_
        )

    def predict_proba(self, X):
        jll = self.predict_joint_log_proba(X)
        return np.exp(jll - logsumexp(jll, axis=1, keepdims=True))

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def build_logit(C: float = 1.0, penalty: str = "l2") -> Pipeline:
    """Familia 1: regresión logística regularizada sobre variables estandarizadas.

    ``l1_ratio`` = 0 equivale a la penalización L2 (Ridge) y ``l1_ratio`` = 1, a L1 (Lasso).
    """
    l1 = penalty == "l1"
    model = LogisticRegression(
        C=C, l1_ratio=1.0 if l1 else 0.0, solver="liblinear" if l1 else "lbfgs", max_iter=5000
    )
    return Pipeline([("pre", build_preprocessor(scale=True)), ("modelo", model)])


def build_naive_bayes() -> Pipeline:
    """Familia 2: Naive Bayes mixto (sin escalado: la gaussiana estima media y varianza)."""
    return Pipeline([("pre", build_preprocessor()), ("modelo", MixedNaiveBayes())])


def build_tree(ccp_alpha: float = 0.0) -> Pipeline:
    """Familia 3: árbol CART. Con ccp_alpha = 0 crece sin restricciones."""
    return Pipeline(
        [
            ("pre", build_preprocessor()),
            ("modelo", DecisionTreeClassifier(ccp_alpha=ccp_alpha, random_state=SEED)),
        ]
    )


def build_forest(**params) -> Pipeline:
    """Familia 4: Random Forest con error out-of-bag habilitado."""
    rf = RandomForestClassifier(oob_score=True, random_state=SEED, n_jobs=-1, **params)
    return Pipeline([("pre", build_preprocessor()), ("modelo", rf)])


def classification_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    """Métricas de discriminación, clasificación y calibración sobre un conjunto dado."""
    pred = (proba >= threshold).astype(int)
    return {
        "AUC-ROC": roc_auc_score(y_true, proba),
        "PR-AUC": average_precision_score(y_true, proba),
        "Brier": brier_score_loss(y_true, proba),
        "Precisión": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred),
        "F1": f1_score(y_true, pred),
        "F1 macro": f1_score(y_true, pred, average="macro"),
        "F1 ponderado": f1_score(y_true, pred, average="weighted"),
    }


def calibration_table(y_true, proba, n_bins: int = 10) -> pd.DataFrame:
    """Frecuencia observada frente a probabilidad media predicha, por decil de probabilidad."""
    obs, pred = calibration_curve(y_true, proba, n_bins=n_bins, strategy="quantile")
    return pd.DataFrame({"prob_predicha": pred, "frecuencia_observada": obs})


def bootstrap_auc_difference(y_true, proba_a, proba_b, n_boot: int = 2000, seed: int = SEED):
    """IC bootstrap del 95 % para AUC(a) − AUC(b), remuestreando pacientes de test."""
    y = np.asarray(y_true)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        diffs.append(roc_auc_score(y[idx], proba_a[idx]) - roc_auc_score(y[idx], proba_b[idx]))
    diffs = np.array(diffs)
    return {
        "diferencia": roc_auc_score(y, proba_a) - roc_auc_score(y, proba_b),
        "IC_inf": np.percentile(diffs, 2.5),
        "IC_sup": np.percentile(diffs, 97.5),
        "p_bilateral": 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()),
    }


def oob_convergence(pipeline: Pipeline, X, y, sizes=(25, 50, 100, 200, 300, 500)) -> pd.DataFrame:
    """Error out-of-bag y AUC out-of-bag a medida que crece la cantidad de árboles."""
    pre = pipeline.named_steps["pre"]
    rf = pipeline.named_steps["modelo"]
    Xt = pre.fit_transform(X)
    rows = []
    for n in sizes:
        model = RandomForestClassifier(**{**rf.get_params(), "n_estimators": n}).fit(Xt, y)
        proba = model.oob_decision_function_[:, 1]
        rows.append({"árboles": n, "error_OOB": 1 - model.oob_score_,
                     "AUC_OOB": roc_auc_score(y, proba)})
    return pd.DataFrame(rows)


def odds_ratio_table(X: pd.DataFrame, y: pd.Series, alpha: float = 0.05) -> pd.DataFrame:
    """Logit por máxima verosimilitud (statsmodels) con odds ratios e IC del 95 %.

    Se estima sin penalización para que los intervalos tengan la interpretación clásica;
    la versión regularizada se usa para predecir.
    """
    fit = sm.Logit(y, sm.add_constant(X.astype(float))).fit(disp=False, maxiter=200)
    ci = fit.conf_int(alpha)
    tab = pd.DataFrame(
        {
            "beta": fit.params,
            "OR": np.exp(fit.params),
            "OR_inf": np.exp(ci[0]),
            "OR_sup": np.exp(ci[1]),
            "z": fit.tvalues,
            "p_valor": fit.pvalues,
        }
    )
    return tab.drop(index="const")
