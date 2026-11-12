"""Web app del TP Integrador: diagnóstico, predicción ("What-Is") y simulación causal ("What-If").

Ejecución local:  streamlit run app/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

import altair as alt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
import ui  # noqa: E402
from ui import HIGHER, LOWER, NO_RHC, PALETTE, RHC, num, pct, pp  # noqa: E402

from rhc import config as cfg  # noqa: E402
from rhc import serving as sv  # noqa: E402
from rhc.inference import wilson_ci  # noqa: E402

REPO = "https://github.com/Valentina-Bolivar/tp-integrador-rhc"

st.set_page_config(
    page_title="Cateterismo cardíaco derecho",
    page_icon="app/static/favicon.png" if (HERE / "static" / "favicon.png").exists() else None,
    layout="wide",
    initial_sidebar_state="collapsed",
)
ui.inject()

LABELS = dict(cfg.LABELS) | {
    "dx": "Diagnóstico principal",
    "sex": "Sexo",
    "race": "Raza",
    "age": "Edad (años)",
}
SEX = {"Female": "Femenino", "Male": "Masculino"}
CANCER = {"No": "Sin cáncer", "Yes": "Localizado", "Metastatic": "Metastásico"}
RACE = {"white": "Blanca", "black": "Negra", "other": "Otra"}
INCOME = {
    "Under $11k": "Menos de USD 11.000",
    "$11-$25k": "USD 11.000 a 25.000",
    "$25-$50k": "USD 25.000 a 50.000",
    "> $50k": "Más de USD 50.000",
}
INSURANCE = {
    "Private": "Privada",
    "Medicare": "Medicare",
    "Private & Medicare": "Privada y Medicare",
    "Medicaid": "Medicaid",
    "Medicare & Medicaid": "Medicare y Medicaid",
    "No insurance": "Sin cobertura",
}
LEVELS = {"sex": SEX, "ca": CANCER, "race": RACE, "income": INCOME, "ninsclas": INSURANCE}
COMORB = [c for c in cfg.COMORBIDITIES if c != "malighx"]
PHYSIO_STEP = {
    "ph1": 0.01,
    "temp1": 0.1,
    "alb1": 0.1,
    "bili1": 0.1,
    "crea1": 0.1,
    "pot1": 0.1,
    "wblc1": 0.1,
}


def label(name: str) -> str:
    return LABELS.get(name, cfg.pretty(name))


def short(name: str) -> str:
    """Etiqueta sin unidades, para ejes y listas."""
    return label(name).split(" (")[0]


# --- Datos y modelos ----------------------------------------------------------------


@st.cache_resource(show_spinner="Cargando datos, modelo y resultados causales")
def artifacts() -> sv.Artifacts:
    return sv.load_artifacts()


art = artifacts()
data, causal = art.data, art.causal
ATE = causal["ate"]
NAIVE = ATE["Ingenuo: Y ~ T"]
LR = ATE["DoWhy: regresión con ajuste backdoor"]
IPW = ATE["DoWhy: IPW (Hájek)"]
DML = causal["ate_dml"]
BASE_MORT = float(data[cfg.OUTCOME_BIN].mean())
YOUDEN = float(art.model_meta["umbral_youden"])
CHK = art.checks
_ages = sorted({int(g["edad"]) for g in causal["cate_grilla"]})
GRID_AGES = ", ".join(str(a) for a in _ages[:-1]) + f" y {_ages[-1]}"
N_PAC = num(len(data), 0)
N_RHC = num(int(data[cfg.TREATMENT].sum()), 0)
N_SENT = num(int(art.audit["casos"].sum()), 0)


# --- Estado del paciente de la pestaña Predicción -----------------------------------


def _set_profile(profile: dict) -> None:
    for c in cfg.NUMERIC_FEATURES:
        value = profile[c]
        st.session_state[f"in_{c}"] = None if value is None or pd.isna(value) else float(value)
    st.session_state["in_surv2md1"] = int(round(100 * (profile["surv2md1"] or 0)))
    for c in cfg.MODEL_CATEGORICAL:
        st.session_state[f"in_{c}"] = profile[c]
    for c in ["dnr1", "has_cat2", "rhc"]:
        st.session_state[f"in_{c}"] = bool(profile[c])
    st.session_state["in_comorb"] = [c for c in COMORB if profile[c] == 1]
    st.session_state["in_admit"] = [c for c in cfg.ADMISSION_DX if profile[c] == 1]
    sync_causal()


def sync_causal() -> None:
    """Lleva diagnóstico, edad y APACHE del paciente a los controles de la pestaña causal."""
    age = st.session_state.get("in_age") or float(data["age"].median())
    aps = st.session_state.get("in_aps1") or float(data["aps1"].median())
    st.session_state["c_dx"] = st.session_state["in_dx"]
    st.session_state["c_age"] = int(np.clip(round(age), 40, 85))
    st.session_state["c_aps"] = int(np.clip(round(aps), 15, 140))


def reset_profile() -> None:
    _set_profile(sv.default_profile(art.train))
    st.session_state["origen"] = None


def random_patient() -> None:
    rng = np.random.default_rng()
    idx = int(rng.integers(0, len(art.test)))
    row = art.test.iloc[idx]
    _set_profile(sv.profile_from_row(row))
    st.session_state["origen"] = {
        "murio": int(row[cfg.OUTCOME_BIN]),
        "rhc": int(row[cfg.TREATMENT]),
        "fila": idx,
    }


def current_profile() -> dict:
    profile = {c: st.session_state[f"in_{c}"] for c in cfg.NUMERIC_FEATURES}
    profile["surv2md1"] = st.session_state["in_surv2md1"] / 100
    profile.update({c: st.session_state[f"in_{c}"] for c in cfg.MODEL_CATEGORICAL})
    profile.update({c: int(st.session_state[f"in_{c}"]) for c in ["dnr1", "has_cat2", "rhc"]})
    profile.update({c: int(c in st.session_state["in_comorb"]) for c in COMORB})
    profile.update({c: int(c in st.session_state["in_admit"]) for c in cfg.ADMISSION_DX})
    return profile


if "in_age" not in st.session_state:
    reset_profile()


# --- Gráficos -----------------------------------------------------------------------

COMMA = "replace(format(datum.value, '{fmt}'), '.', ',')"


def axis(title: str | None, fmt: str = ".2~f", **kw) -> alt.Axis:
    return alt.Axis(title=title, labelExpr=COMMA.format(fmt=fmt), **kw)


def estimators_chart(compact: bool = False) -> alt.Chart:
    rows = [
        ("Diferencia cruda", NAIVE, "cruda"),
        ("Regresión con ajuste", LR, "ajustada"),
        ("Ponderación por propensión", IPW, "ajustada"),
        ("Double Machine Learning", DML, "ajustada"),
    ]
    df = pd.DataFrame(
        [
            {
                "estimador": name,
                "valor": 100 * r["valor"],
                "lo": 100 * r["ic_inf"],
                "hi": 100 * r["ic_sup"],
                "tipo": kind,
                "texto": f"{num(100 * r['valor'])} pp",
                "tip": f"{num(100 * r['valor'])} pp (IC 95 %: {num(100 * r['ic_inf'])} a "
                f"{num(100 * r['ic_sup'])})",
            }
            for name, r, kind in rows
        ]
    )
    order = df["estimador"].tolist()
    color = alt.Color(
        "tipo:N",
        scale=alt.Scale(domain=["cruda", "ajustada"], range=[PALETTE["faint"], PALETTE["ink"]]),
        legend=None,
    )
    y = alt.Y(
        "estimador:N",
        sort=order,
        title=None,
        axis=alt.Axis(labelLimit=260, labelFontSize=14 if not compact else 13),
    )
    base = alt.Chart(df).encode(y=y)
    x_scale = alt.Scale(domain=[-1, 13])
    rule = base.mark_rule(strokeWidth=2.5, strokeCap="round").encode(
        x=alt.X("lo:Q", scale=x_scale, axis=axis("Cambio en la mortalidad a 30 días (pp)")),
        x2="hi:Q",
        color=color,
        tooltip=[
            alt.Tooltip("estimador:N", title="Estimador"),
            alt.Tooltip("tip:N", title="Efecto"),
        ],
    )
    dots = base.mark_point(filled=True, size=120, opacity=1).encode(
        x="valor:Q",
        color=color,
        tooltip=[
            alt.Tooltip("estimador:N", title="Estimador"),
            alt.Tooltip("tip:N", title="Efecto"),
        ],
    )
    text = base.mark_text(align="left", dx=10, fontSize=13).encode(
        x="hi:Q", text="texto:N", color=color
    )
    zero = (
        alt.Chart(pd.DataFrame({"x": [0]}))
        .mark_rule(color=PALETTE["muted"], strokeDash=[3, 3])
        .encode(x="x:Q")
    )
    return (zero + rule + dots + text).properties(height=190 if compact else 230)


def mortality_by_group(df: pd.DataFrame) -> alt.Chart:
    rows = []
    for t, name in [(0, "Sin RHC"), (1, "Con RHC")]:
        g = df[df[cfg.TREATMENT] == t]
        if len(g) == 0:
            continue
        k = int(g[cfg.OUTCOME_BIN].sum())
        lo, hi = wilson_ci(k, len(g))
        rows.append(
            {
                "grupo": name,
                "tasa": 100 * k / len(g),
                "lo": 100 * lo,
                "hi": 100 * hi,
                "n": len(g),
                "texto": pct(k / len(g)),
                "tip": f"{pct(k / len(g))} de {num(len(g), 0)} pacientes",
            }
        )
    d = pd.DataFrame(rows)
    color = alt.Color(
        "grupo:N", scale=alt.Scale(domain=["Sin RHC", "Con RHC"], range=[NO_RHC, RHC]), legend=None
    )
    base = alt.Chart(d).encode(
        y=alt.Y(
            "grupo:N",
            title=None,
            sort=["Sin RHC", "Con RHC"],
            axis=alt.Axis(labelFontSize=14, labelPadding=8),
        )
    )
    bars = base.mark_bar(cornerRadiusEnd=4, height=26).encode(
        x=alt.X("tasa:Q", axis=axis("Mortalidad a 30 días (%)"), scale=alt.Scale(domain=[0, 100])),
        color=color,
        tooltip=[alt.Tooltip("tip:N", title="Mortalidad")],
    )
    ci = base.mark_rule(color=PALETTE["ink"], strokeWidth=1.5).encode(x="lo:Q", x2="hi:Q")
    text = base.mark_text(align="left", dx=8, fontSize=13).encode(x="hi:Q", text="texto:N")
    return (bars + ci + text).properties(height=130)


def dumbbell_by_dx(df: pd.DataFrame) -> alt.Chart:
    rows = []
    for dx, g in df.groupby("dx", observed=True):
        for t, name in [(0, "Sin RHC"), (1, "Con RHC")]:
            h = g[g[cfg.TREATMENT] == t]
            if len(h) >= 10:
                rows.append(
                    {
                        "dx": dx,
                        "grupo": name,
                        "tasa": 100 * h[cfg.OUTCOME_BIN].mean(),
                        "n": len(h),
                        "rhc_pct": 100 * g[cfg.TREATMENT].mean(),
                        "tip": f"{pct(h[cfg.OUTCOME_BIN].mean())} de {num(len(h), 0)}",
                    }
                )
    d = pd.DataFrame(rows)
    if d.empty:
        return None
    order = d.groupby("dx")["rhc_pct"].first().sort_values(ascending=False).index.tolist()
    d["etiqueta"] = d.apply(lambda r: f"{r['dx']} ({num(r['rhc_pct'], 0)} % con RHC)", axis=1)
    lab_order = [d.loc[d["dx"] == x, "etiqueta"].iloc[0] for x in order]
    y = alt.Y("etiqueta:N", sort=lab_order, title=None, axis=alt.Axis(labelLimit=320))
    line = (
        alt.Chart(d)
        .mark_rule(color=PALETTE["rule"], strokeWidth=3)
        .encode(
            y=y,
            x=alt.X(
                "min(tasa):Q",
                axis=axis("Mortalidad a 30 días (%)"),
                scale=alt.Scale(domain=[0, 80]),
            ),
            x2="max(tasa):Q",
        )
    )
    dots = (
        alt.Chart(d)
        .mark_point(filled=True, size=110, opacity=1)
        .encode(
            y=y,
            x="tasa:Q",
            color=alt.Color(
                "grupo:N",
                scale=alt.Scale(domain=["Sin RHC", "Con RHC"], range=[NO_RHC, RHC]),
                legend=alt.Legend(title=None),
            ),
            tooltip=[
                alt.Tooltip("dx:N", title="Diagnóstico"),
                alt.Tooltip("grupo:N", title="Grupo"),
                alt.Tooltip("tip:N", title="Mortalidad"),
            ],
        )
    )
    return (line + dots).properties(height=34 * len(order) + 40)


# --- Encabezado ---------------------------------------------------------------------

per_1000 = round(1000 * LR["valor"])
st.markdown(
    """
<div class="masthead">
<p class="kicker">Trabajo Práctico Integrador de Estadística Aplicada y Modelos de Regresión,
Universidad Católica Argentina, 2026</p>
<h1>Cateterismo cardíaco derecho en terapia intensiva</h1>
</div>
""",
    unsafe_allow_html=True,
)
head_l, head_r = st.columns([1.15, 1], gap="large")
with head_l:
    st.markdown(
        f"""
<div class="masthead" style="border:none;padding:0;margin:0.4rem 0 0 0">
<p class="lead">Aplicado de forma rutinaria, el cateterismo agrega alrededor de
<strong>{per_1000} muertes cada 1.000 pacientes</strong> a los 30 días (IC 95 %:
{round(1000 * LR["ic_inf"])} a {round(1000 * LR["ic_sup"])}). La comparación cruda, de
{round(1000 * NAIVE["valor"])} cada 1.000, exagera el daño porque los pacientes más graves son
los que con más frecuencia lo reciben.</p>
<p class="note">{num(len(data), 0)} pacientes críticos del estudio SUPPORT (Connors et al., 1996).
Esta aplicación permite explorar la muestra, estimar el riesgo de un paciente nuevo y simular qué
ocurre con la mortalidad si se decide cateterizar.</p>
</div>
""",
        unsafe_allow_html=True,
    )
with head_r:
    st.markdown(
        "<p class='caption' style='margin:0.4rem 0 0 0'>Efecto del cateterismo sobre la "
        "mortalidad a 30 días según cuatro estimadores, con su intervalo de confianza del "
        "95 %.</p>",
        unsafe_allow_html=True,
    )
    st.altair_chart(estimators_chart(compact=True), width="stretch", theme=None)

tab_dx, tab_pred, tab_causal, tab_about = st.tabs(
    ["Diagnóstico y exploración", "Predicción del riesgo", "Simulación causal", "Sobre el trabajo"]
)

# ====================================================================================
# 1. Diagnóstico y exploración
# ====================================================================================
with tab_dx:
    ui.section(
        "Explorar la muestra",
        "Los filtros se aplican a todos los gráficos de esta pestaña. Las cifras describen a los "
        "pacientes observados: no separan el efecto del tratamiento de la gravedad de quien lo "
        "recibe.",
    )
    f1, f2, f3, f4 = st.columns([1.25, 1, 1, 1.15], gap="medium")
    with f1:
        dx_sel = st.multiselect("Diagnóstico principal", cfg.DX_ORDER, placeholder="Todos")
    with f2:
        aps_lo, aps_hi = int(data["aps1"].min()), int(data["aps1"].max())
        aps_rng = st.slider("Score APACHE III", aps_lo, aps_hi, (aps_lo, aps_hi))
    with f3:
        age_lo, age_hi = int(np.floor(data["age"].min())), int(np.ceil(data["age"].max()))
        age_rng = st.slider("Edad (años)", age_lo, age_hi, (age_lo, age_hi))
    with f4:
        trt = (
            st.segmented_control(
                "Tratamiento", ["Todos", "Con RHC", "Sin RHC"], default="Todos", key="f_trt"
            )
            or "Todos"
        )

    mask = data["aps1"].between(*aps_rng) & data["age"].between(*age_rng)
    if dx_sel:
        mask &= data["dx"].isin(dx_sel)
    if trt != "Todos":
        mask &= data[cfg.TREATMENT] == (1 if trt == "Con RHC" else 0)
    view = data[mask]

    if view.empty:
        st.info(
            "Ningún paciente cumple los filtros elegidos. Amplíe el rango de APACHE o de edad, "
            "o quite algún diagnóstico."
        )
    else:
        mort = view[cfg.OUTCOME_BIN].mean()
        k1, k2, k3, k4 = st.columns(4)
        k1.metric(
            "Pacientes",
            num(len(view), 0),
            border=True,
            delta=f"{pct(len(view) / len(data), 0)} de la muestra",
            delta_color="off",
            delta_arrow="off",
        )
        same = abs(mort - BASE_MORT) < 0.0005
        k2.metric(
            "Mortalidad a 30 días",
            pct(mort),
            border=True,
            delta="igual a la muestra completa"
            if same
            else f"{pp(mort - BASE_MORT)} frente al total",
            delta_color="off" if same else "inverse",
            delta_arrow="off" if same else "auto",
        )
        k3.metric(
            "Recibieron RHC",
            pct(view[cfg.TREATMENT].mean()),
            border=True,
            delta=f"{num(int(view[cfg.TREATMENT].sum()), 0)} pacientes",
            delta_color="off",
            delta_arrow="off",
        )
        k4.metric(
            "APACHE III mediano",
            num(view["aps1"].median(), 0),
            border=True,
            delta=f"rango intercuartil {num(view['aps1'].quantile(0.25), 0)} a "
            f"{num(view['aps1'].quantile(0.75), 0)}",
            delta_color="off",
            delta_arrow="off",
        )

        st.space("small")
        c1, c2 = st.columns([1, 1.25], gap="large")
        with c1:
            st.markdown("#### Mortalidad observada por grupo")
            st.altair_chart(mortality_by_group(view), width="stretch", theme=None)
            if view[cfg.TREATMENT].nunique() == 2:
                diff = (
                    view.loc[view[cfg.TREATMENT] == 1, cfg.OUTCOME_BIN].mean()
                    - view.loc[view[cfg.TREATMENT] == 0, cfg.OUTCOME_BIN].mean()
                )
                ui.caption(
                    f"Diferencia cruda: {pp(diff)}. Barras con intervalo de Wilson del 95 %. Esta "
                    "diferencia mezcla el efecto del procedimiento con la confusión por "
                    "indicación; la estimación causal está en la pestaña Simulación causal."
                )
        with c2:
            st.markdown("#### Mortalidad por diagnóstico y tratamiento")
            chart = dumbbell_by_dx(view)
            if chart is not None:
                st.altair_chart(chart, width="stretch", theme=None)
            ui.caption(
                "Los diagnósticos se ordenan según la proporción de pacientes que recibió "
                "RHC. Se omiten los grupos con menos de 10 pacientes."
            )

        # --- Inspector de variables --------------------------------------------------
        st.divider()
        ui.section(
            "Inspeccionar una variable",
            "Distribución de una variable en dos grupos, con sus estadísticos y la "
            "diferencia de medias estandarizada. Una diferencia mayor que 0,1 en valor "
            "absoluto indica desbalance entre los grupos.",
        )
        variables = (
            cfg.NUMERIC_FEATURES + COMORB + cfg.ADMISSION_DX + ["dnr1"] + cfg.MODEL_CATEGORICAL
        )
        i1, i2 = st.columns([1.4, 1], gap="medium")
        with i1:
            var = st.selectbox(
                "Variable", variables, index=variables.index("aps1"), format_func=label
            )
        with i2:
            split_by = (
                st.segmented_control(
                    "Comparar según",
                    ["Tratamiento", "Resultado a 30 días"],
                    default="Tratamiento",
                    key="f_split",
                )
                or "Tratamiento"
            )
        if split_by == "Tratamiento":
            gcol, gmap, gcolors = cfg.TREATMENT, {0: "Sin RHC", 1: "Con RHC"}, [NO_RHC, RHC]
        else:
            gcol, gmap, gcolors = cfg.OUTCOME_BIN, {0: "Sobrevivió", 1: "Murió"}, [LOWER, HIGHER]
        groups = list(gmap.values())
        gscale = alt.Scale(domain=groups, range=gcolors)

        v1, v2 = st.columns([1.5, 1], gap="large")
        if var in cfg.NUMERIC_FEATURES:
            x = view[var].dropna()
            lo_q, hi_q = x.quantile(0.005), x.quantile(0.995)
            edges = np.linspace(lo_q, hi_q, 31)
            rows = []
            for code, name in gmap.items():
                vals = view.loc[view[gcol] == code, var].dropna().clip(lo_q, hi_q)
                if len(vals) == 0:
                    continue
                counts, _ = np.histogram(vals, bins=edges)
                share = counts / counts.sum() * 100
                for a, s in zip(edges[:-1], share, strict=True):
                    rows.append({"x": a, "pct": s, "grupo": name})
                rows.append({"x": edges[-1], "pct": share[-1], "grupo": name})
            h = pd.DataFrame(rows)
            hist_legend = alt.Legend(title=None, symbolOpacity=1, symbolType="square")
            base = alt.Chart(h).encode(
                x=alt.X(
                    "x:Q", title=label(var), axis=axis(label(var), ".1f" if hi_q < 20 else ".0f")
                ),
                y=alt.Y("pct:Q", axis=axis("Pacientes del grupo (%)"), stack=None),
                color=alt.Color("grupo:N", scale=gscale, legend=hist_legend),
            )
            hist = base.mark_area(interpolate="step-after", opacity=0.14) + base.mark_line(
                interpolate="step-after", strokeWidth=2
            )
            with v1:
                st.altair_chart(hist.properties(height=300), width="stretch", theme=None)
                ui.caption(
                    "Porcentaje de pacientes de cada grupo en cada intervalo. Se recortan el "
                    "0,5 % inferior y superior para que las colas no aplasten el gráfico."
                )
            with v2:
                stats = []
                for code, name in gmap.items():
                    g = view.loc[view[gcol] == code, var]
                    if len(g) == 0:
                        continue
                    stats.append((name, g))
                st.markdown(f"#### {short(var)}")
                rows = [
                    ("Pacientes", *[num(len(g), 0) for _, g in stats]),
                    ("Datos faltantes", *[pct(g.isna().mean()) for _, g in stats]),
                    ("Media", *[num(g.mean(), 2) for _, g in stats]),
                    ("Mediana", *[num(g.median(), 2) for _, g in stats]),
                    ("Desvío estándar", *[num(g.std(), 2) for _, g in stats]),
                    (
                        "Percentiles 5 y 95",
                        *[
                            f"{num(g.quantile(0.05), 1)} a {num(g.quantile(0.95), 1)}"
                            for _, g in stats
                        ],
                    ),
                ]
                table = pd.DataFrame(rows, columns=["Estadístico", *[n for n, _ in stats]])
                st.dataframe(table, hide_index=True, width="stretch")
                if len(stats) == 2:
                    d = sv.smd(stats[1][1], stats[0][1])
                    verdict = "desbalance relevante" if abs(d) > 0.1 else "grupos balanceados"
                    ui.facts([("Diferencia de medias estandarizada", f"{num(d, 2)} ({verdict})")])
        else:
            if var in cfg.MODEL_CATEGORICAL:
                levels = cfg.CATEGORY_LEVELS[var]
                mapping = LEVELS.get(var, {lv: lv for lv in levels})
                rows = []
                for code, name in gmap.items():
                    g = view.loc[view[gcol] == code, var]
                    if len(g) == 0:
                        continue
                    share = g.value_counts(normalize=True)
                    for lv in levels:
                        rows.append(
                            {
                                "nivel": mapping.get(lv, lv),
                                "grupo": name,
                                "pct": 100 * share.get(lv, 0.0),
                            }
                        )
                order = [mapping.get(lv, lv) for lv in levels]
            else:
                rows = []
                for code, name in gmap.items():
                    g = view.loc[view[gcol] == code, var]
                    if len(g):
                        rows.append({"nivel": "Presente", "grupo": name, "pct": 100 * g.mean()})
                order = ["Presente"]
            d = pd.DataFrame(rows)
            d["texto"] = d["pct"].map(lambda v: f"{num(v)} %")
            chart = alt.Chart(d).encode(
                y=alt.Y("nivel:N", sort=order, title=None, axis=alt.Axis(labelLimit=260)),
                yOffset=alt.YOffset("grupo:N", sort=groups),
                x=alt.X("pct:Q", axis=axis("Pacientes del grupo (%)")),
                color=alt.Color("grupo:N", scale=gscale, legend=alt.Legend(title=None)),
            )
            bars = chart.mark_bar(cornerRadiusEnd=4, height=14) + chart.mark_text(
                align="left", dx=6, fontSize=12, color=PALETTE["ink"]
            ).encode(text="texto:N")
            with v1:
                st.altair_chart(
                    bars.properties(height=max(120, 46 * len(order) + 40)),
                    width="stretch",
                    theme=None,
                )
            with v2:
                st.markdown(f"#### {short(var)}")
                if var not in cfg.MODEL_CATEGORICAL:
                    a = view.loc[view[gcol] == 1, var]
                    b = view.loc[view[gcol] == 0, var]
                    if len(a) and len(b):
                        ui.facts(
                            [
                                (f"Prevalencia, {groups[1].lower()}", pct(a.mean())),
                                (f"Prevalencia, {groups[0].lower()}", pct(b.mean())),
                                ("Diferencia de medias estandarizada", num(sv.smd(a, b), 2)),
                            ]
                        )
                else:
                    ui.caption(
                        "Distribución de los niveles dentro de cada grupo. La primera "
                        "categoría es la referencia de los modelos."
                    )

        # --- PCA -----------------------------------------------------------------------
        st.divider()
        load = sv.pca_loadings(art)
        ev = load.attrs["varianza"]
        ui.section(
            "Mapa de pacientes en el plano de componentes principales",
            f"El PCA se ajustó solo con el conjunto de entrenamiento, sobre las "
            f"{len(cfg.NUMERIC_FEATURES)} variables numéricas estandarizadas, y aquí se proyectan "
            f"los pacientes filtrados. El primer componente explica el {num(100 * ev[0])} % de la "
            f"varianza y el segundo, el {num(100 * ev[1])} %.",
        )
        p1, p2 = st.columns([1.7, 1], gap="large")
        with p2:
            color_by = (
                st.segmented_control(
                    "Color de los puntos",
                    ["Resultado", "Tratamiento", "APACHE III"],
                    default="Resultado",
                    key="f_pca",
                )
                or "Resultado"
            )
            comp = (
                st.segmented_control(
                    "Cargas del componente", ["CP1", "CP2"], default="CP1", key="f_comp"
                )
                or "CP1"
            )
            top = load[comp].reindex(load[comp].abs().sort_values(ascending=False).index)[:8]
            ld = pd.DataFrame({"var": [short(v) for v in top.index], "carga": top.values})
            ld["texto"] = ld["carga"].map(lambda v: num(v, 2))
            lchart = alt.Chart(ld).encode(
                y=alt.Y(
                    "var:N", sort=ld["var"].tolist(), title=None, axis=alt.Axis(labelLimit=220)
                ),
                x=alt.X(
                    "carga:Q",
                    axis=axis(f"Carga en {comp}", ".1f"),
                    scale=alt.Scale(domain=[-0.6, 0.6]),
                ),
            )
            lbars = lchart.mark_bar(cornerRadiusEnd=3, height=14, color=PALETTE["accent"])
            zero = (
                alt.Chart(pd.DataFrame({"x": [0]}))
                .mark_rule(color=PALETTE["muted"])
                .encode(x="x:Q")
            )
            st.altair_chart((lbars + zero).properties(height=260), width="stretch", theme=None)
            pos = [short(v) for v in top.index[:3]]
            ui.caption(
                f"Las variables con mayor peso en {comp} son {pos[0]}, "
                f"{pos[1]} y {pos[2]}. El signo de un componente es "
                "arbitrario: lo que importa es qué variables se mueven juntas."
            )
        with p1:
            sc = sv.pca_scores(art, view).join(
                view[["age", "aps1", "dx", cfg.TREATMENT, cfg.OUTCOME_BIN]]
            )
            sc["Resultado"] = sc[cfg.OUTCOME_BIN].map({0: "Sobrevivió", 1: "Murió"})
            sc["Tratamiento"] = sc[cfg.TREATMENT].map({0: "Sin RHC", 1: "Con RHC"})
            sc["Edad"] = sc["age"].round(0)
            sc["APACHE III"] = sc["aps1"]
            sc["Diagnóstico"] = sc["dx"].astype(str)
            if color_by == "Resultado":
                color = alt.Color(
                    "Resultado:N",
                    scale=alt.Scale(domain=["Sobrevivió", "Murió"], range=[LOWER, HIGHER]),
                    legend=alt.Legend(title=None),
                )
            elif color_by == "Tratamiento":
                color = alt.Color(
                    "Tratamiento:N",
                    scale=alt.Scale(domain=["Sin RHC", "Con RHC"], range=[NO_RHC, RHC]),
                    legend=alt.Legend(title=None),
                )
            else:
                color = alt.Color(
                    "APACHE III:Q",
                    scale=alt.Scale(range=["#D6E6F2", "#123E5E"]),
                    legend=alt.Legend(title="APACHE III", gradientLength=160),
                )
            pts = (
                alt.Chart(sc)
                .mark_circle(size=22, opacity=0.6, stroke="white", strokeWidth=0.4)
                .encode(
                    x=alt.X("CP1:Q", axis=axis("Componente principal 1", tickMinStep=1)),
                    y=alt.Y("CP2:Q", axis=axis("Componente principal 2", tickMinStep=1)),
                    color=color,
                    tooltip=[
                        "Diagnóstico:N",
                        "Edad:Q",
                        "APACHE III:Q",
                        "Tratamiento:N",
                        "Resultado:N",
                    ],
                )
            )
            layers = [pts]
            me = sv.profile_frame(current_profile())
            me_sc = sv.pca_scores(art, me)
            me_sc["nombre"] = "Paciente de la pestaña Predicción"
            layers.append(
                alt.Chart(me_sc)
                .mark_point(
                    shape="diamond",
                    size=300,
                    filled=True,
                    color=PALETTE["ink"],
                    stroke="white",
                    strokeWidth=2,
                    opacity=1,
                )
                .encode(x="CP1:Q", y="CP2:Q", tooltip=[alt.Tooltip("nombre:N", title="")])
            )
            layers.append(
                alt.Chart(me_sc)
                .mark_text(
                    align="left",
                    dx=12,
                    dy=-10,
                    fontSize=13,
                    fontStyle="italic",
                    color=PALETTE["ink"],
                )
                .encode(x="CP1:Q", y="CP2:Q", text="nombre:N")
            )
            st.altair_chart(
                alt.layer(*layers).properties(height=470).interactive(), width="stretch", theme=None
            )
            ui.caption(
                "Cada punto es un paciente. El rombo negro es el paciente que se está "
                "evaluando en la pestaña Predicción del riesgo, proyectado con el mismo PCA. "
                "El gráfico admite zoom y desplazamiento."
            )

        # --- Calidad de datos -----------------------------------------------------------
        st.divider()
        with st.expander("Calidad de los datos: valores centinela corregidos en la auditoría"):
            audit = art.audit.copy()
            audit["variable"] = audit["variable"].map(label)
            audit = audit.rename(
                columns={
                    "variable": "Variable",
                    "regla": "Regla",
                    "casos": "Casos",
                    "media_cruda": "Media cruda",
                    "media_limpia": "Media corregida",
                    "justificación": "Justificación",
                }
            )
            st.dataframe(
                audit[
                    [
                        "Variable",
                        "Regla",
                        "Casos",
                        "Media cruda",
                        "Media corregida",
                        "Justificación",
                    ]
                ],
                hide_index=True,
                width="stretch",
                column_config={
                    "Media cruda": st.column_config.NumberColumn(format="%.2f"),
                    "Media corregida": st.column_config.NumberColumn(format="%.2f"),
                },
            )
            ui.caption(
                f"En total se reemplazaron {num(int(art.audit['casos'].sum()), 0)} valores "
                "imposibles por faltantes. Los ceros de la escala de coma, la diuresis y "
                "los años de educación son legítimos y se conservan. El detalle está en la "
                "sección 1.2 del notebook."
            )
        cols = ["dx", "age", "sex", "aps1", cfg.TREATMENT, cfg.OUTCOME_BIN] + [
            c for c in cfg.NUMERIC_FEATURES if c not in ("age", "aps1")
        ]
        st.download_button(
            "Descargar los pacientes filtrados (CSV)",
            view[cols].to_csv(index=False).encode("utf-8"),
            file_name="rhc_filtrado.csv",
            mime="text/csv",
        )

# ====================================================================================
# 2. Predicción del riesgo ("What-Is")
# ====================================================================================
with tab_pred:
    meta = art.model_meta["metricas_test"]
    ui.section(
        "Riesgo de muerte a 30 días de un paciente nuevo",
        f"Modelo final del torneo de la Fase 3: regresión logística con penalización L2 "
        f"(C = {num(art.model_meta['C'], 1)}). En el conjunto de test obtuvo un AUC-ROC de "
        f"{num(meta['AUC-ROC'], 3)} y un puntaje de Brier de {num(meta['Brier'], 3)}. El resultado "
        "es una predicción del riesgo, no una recomendación de tratamiento.",
    )
    b1, b2, _ = st.columns([1, 1.25, 2.2])
    b1.button(
        "Paciente de referencia",
        on_click=reset_profile,
        width="stretch",
        help="Valores medianos, sin comorbilidades y en las categorías más frecuentes.",
    )
    b2.button(
        "Paciente real al azar",
        on_click=random_patient,
        width="stretch",
        type="primary",
        help="Carga un paciente del conjunto de test, que el modelo no vio al entrenarse.",
    )

    origin = st.session_state.get("origen")
    left, right = st.columns([1, 1.25], gap="large")
    with left:
        with st.container(border=True):
            st.markdown("#### Perfil clínico")
            a1, a2 = st.columns(2)
            a1.number_input(
                "Edad (años)",
                18.0,
                105.0,
                step=1.0,
                format="%.0f",
                key="in_age",
                on_change=sync_causal,
            )
            a2.selectbox("Sexo", cfg.CATEGORY_LEVELS["sex"], format_func=SEX.get, key="in_sex")
            st.selectbox("Diagnóstico principal", cfg.DX_ORDER, key="in_dx", on_change=sync_causal)
            a3, a4 = st.columns(2)
            a3.selectbox("Cáncer", cfg.CATEGORY_LEVELS["ca"], format_func=CANCER.get, key="in_ca")
            a4.toggle("Diagnóstico secundario", key="in_has_cat2")
            st.slider(
                "Score APACHE III",
                0.0,
                150.0,
                step=1.0,
                format="%.0f",
                key="in_aps1",
                on_change=sync_causal,
                help="Gravedad fisiológica aguda; a mayor valor, mayor gravedad.",
            )
            st.slider(
                "Probabilidad de sobrevida a 2 meses (modelo SUPPORT)",
                0,
                100,
                step=1,
                format="%d %%",
                key="in_surv2md1",
            )
            a5, a6 = st.columns(2)
            a5.slider(
                "Escala de coma",
                0.0,
                100.0,
                step=1.0,
                format="%.0f",
                key="in_scoma1",
                help="0 indica paciente consciente.",
            )
            a6.slider(
                "Índice DASI",
                10.0,
                33.0,
                step=0.5,
                format="%.1f",
                key="in_das2d3pc",
                help="Capacidad funcional previa; a mayor valor, mejor capacidad.",
            )
            a7, a8 = st.columns(2)
            a7.toggle("Orden de no reanimar", key="in_dnr1")
            a8.toggle("RHC en las primeras 24 h", key="in_rhc")

        with st.expander("Fisiología del día 1"):
            ui.caption(
                "Un campo vacío indica un dato no registrado: el modelo lo imputa con la "
                "mediana de entrenamiento y, para peso, presión, frecuencia cardíaca y "
                "respiratoria, activa además su indicador de faltante."
            )
            phys = [c for c in cfg.PHYSIOLOGY]
            cols3 = st.columns(3)
            for i, c in enumerate(phys):
                step = PHYSIO_STEP.get(c, 1.0)
                cols3[i % 3].number_input(
                    label(c),
                    min_value=0.0,
                    step=step,
                    format="%.2f" if step < 0.1 else ("%.1f" if step < 1 else "%.0f"),
                    key=f"in_{c}",
                    placeholder="no registrado",
                )
        with st.expander("Comorbilidades y motivo de admisión"):
            st.multiselect(
                "Comorbilidades", COMORB, format_func=label, key="in_comorb", placeholder="Ninguna"
            )
            st.multiselect(
                "Motivos de admisión",
                cfg.ADMISSION_DX,
                format_func=label,
                key="in_admit",
                placeholder="Ninguno",
            )
        with st.expander("Datos sociodemográficos"):
            s1, s2 = st.columns(2)
            s1.number_input("Educación (años)", 0.0, 30.0, step=1.0, format="%.0f", key="in_edu")
            s2.selectbox("Raza", cfg.CATEGORY_LEVELS["race"], format_func=RACE.get, key="in_race")
            st.selectbox(
                "Ingreso anual",
                cfg.CATEGORY_LEVELS["income"],
                format_func=INCOME.get,
                key="in_income",
            )
            st.selectbox(
                "Cobertura médica",
                cfg.CATEGORY_LEVELS["ninsclas"],
                format_func=INSURANCE.get,
                key="in_ninsclas",
            )

    profile = current_profile()
    pred = sv.predict(art, profile)
    p = pred["p"]

    with right:
        card = st.container()
        threshold = st.slider(
            "Umbral de clasificación",
            5,
            80,
            round(100 * YOUDEN),
            1,
            format="%d %%",
            key="thr",
            help=f"El umbral de Youden ({pct(YOUDEN, 0)}) maximiza la suma de sensibilidad y "
            "especificidad en validación cruzada. Bajarlo detecta más muertes a "
            "costa de más falsas alarmas.",
        )
        threshold = threshold / 100
        high = p >= threshold
        verdict = (
            f"Supera el umbral de {pct(threshold, 0)}: el paciente se clasifica como de "
            "alto riesgo."
            if high
            else f"Está por debajo del umbral de {pct(threshold, 0)}: el paciente no se "
            "clasifica como de alto riesgo."
        )
        ticks = "".join(
            f'<span class="tick{" first" if t == 0 else " last" if t == 100 else ""}" '
            f'style="left:{t}%">{t} %</span>'
            for t in (0, 25, 50, 75, 100)
        )
        origin_txt = ""
        if origin:
            outcome = "murió" if origin["murio"] else "sobrevivió"
            treated = "recibió" if origin["rhc"] else "no recibió"
            origin_txt = (
                f'<p class="caption" style="margin-top:0.6rem">Paciente real del conjunto '
                f"de test: {treated} RHC y {outcome} a los 30 días. Si se modifica algún "
                "valor, el perfil deja de corresponder a ese paciente.</p>"
            )
        card.markdown(
            f"""
<div class="riskcard">
<p class="label">Probabilidad estimada de muerte a 30 días</p>
<p class="value">{num(100 * p, 0)}<small> %</small></p>
<p class="ci">Intervalo de confianza del 95 %: {pct(pred["ic_inf"], 0)} a {pct(pred["ic_sup"], 0)}
(bootstrap del modelo, {sv.N_BOOT} réplicas). Mortalidad media de la muestra: {pct(BASE_MORT, 0)}.
</p>
<div class="riskbar">
  <div class="band" style="left:{100 * pred["ic_inf"]:.2f}%;width:{
                100 * (pred["ic_sup"] - pred["ic_inf"]):.2f}%"></div>
  <div class="thr" style="left:{100 * threshold:.2f}%"><span>umbral</span></div>
  <div class="dot" style="left:{100 * p:.2f}%"></div>
  {ticks}
</div>
<p class="verdict">{verdict}</p>
{origin_txt}
</div>
""",
            unsafe_allow_html=True,
        )
        y_test = art.test[cfg.OUTCOME_BIN].to_numpy()
        pred_pos = art.test_proba >= threshold
        tp = int((pred_pos & (y_test == 1)).sum())
        sens = tp / max(int((y_test == 1).sum()), 1)
        spec = int((~pred_pos & (y_test == 0)).sum()) / max(int((y_test == 0).sum()), 1)
        ppv = tp / max(int(pred_pos.sum()), 1)
        ui.caption(
            f"Con este umbral, en el conjunto de test el modelo detecta el {pct(sens, 0)} de "
            f"las muertes (sensibilidad), descarta correctamente al {pct(spec, 0)} de los "
            f"sobrevivientes (especificidad) y acierta en el {pct(ppv, 0)} de los pacientes "
            "que marca como de alto riesgo."
        )

        # --- Contribuciones -------------------------------------------------------------
        st.markdown("#### Qué explica este riesgo")
        contrib = sv.contributions(art, profile)
        top = contrib.head(10)

        def value_text(c: str) -> str:
            if c in cfg.MODEL_CATEGORICAL:
                lv = profile[c]
                return LEVELS.get(c, {}).get(lv, lv)
            if c in cfg.NUMERIC_FEATURES:
                v = profile[c]
                if v is None or pd.isna(v):
                    return "no registrado"
                if c == "surv2md1":
                    return pct(v, 0)
                return num(v, 0 if float(v).is_integer() or v >= 10 else 2)
            return "sí" if profile.get(c, 0) else "no"

        cd = pd.DataFrame(
            {
                "var": [f"{short(c)}: {value_text(c)}" for c in top.index],
                "aporte": top.values,
            }
        )
        cd["sentido"] = np.where(cd["aporte"] > 0, "Aumenta el riesgo", "Reduce el riesgo")
        cd["texto"] = cd["aporte"].map(lambda v: f"×{num(np.exp(v), 2)}")
        lim = max(0.5, float(np.abs(cd["aporte"]).max()) * 1.25)
        cchart = alt.Chart(cd).encode(
            y=alt.Y("var:N", sort=cd["var"].tolist(), title=None, axis=alt.Axis(labelLimit=300)),
            x=alt.X(
                "aporte:Q",
                scale=alt.Scale(domain=[-lim, lim]),
                axis=axis("Aporte al log-odds de muerte", ".1f"),
            ),
        )
        cbars = cchart.mark_bar(cornerRadiusEnd=3, height=15).encode(
            color=alt.Color(
                "sentido:N",
                scale=alt.Scale(
                    domain=["Aumenta el riesgo", "Reduce el riesgo"], range=[HIGHER, LOWER]
                ),
                legend=alt.Legend(title=None),
            ),
            tooltip=[
                alt.Tooltip("var:N", title="Variable"),
                alt.Tooltip("texto:N", title="Multiplica las chances por"),
            ],
        )
        ctext = cchart.mark_text(fontSize=12, color=PALETTE["muted"], align="left", dx=6).encode(
            text="texto:N"
        ).transform_filter("datum.aporte > 0") + cchart.mark_text(
            fontSize=12, color=PALETTE["muted"], align="right", dx=-6
        ).encode(text="texto:N").transform_filter("datum.aporte <= 0")
        czero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=PALETTE["ink"]).encode(x="x:Q")
        st.altair_chart(
            (cbars + ctext + czero).properties(height=34 * len(cd) + 40),
            width="stretch",
            theme=None,
        )
        ui.caption(
            "Cada barra compara a este paciente con uno de valores promedio, sin "
            "comorbilidades y en las categorías de referencia. El número indica por cuánto "
            "multiplica la variable las chances de morir. Son las diez variables de mayor "
            "aporte; la explicación describe al modelo, no relaciones causales."
        )

        # --- Calibración ----------------------------------------------------------------
        st.markdown("#### Cuánto creerle a esta probabilidad")
        cal = sv.calibration_bins(art)
        for c in ["predicha", "observada", "ic_inf", "ic_sup"]:
            cal[c] = 100 * cal[c]
        cal["tip"] = cal.apply(
            lambda r: (
                f"predicha {num(r['predicha'])} %, observada "
                f"{num(r['observada'])} % ({r['n']} pacientes)"
            ),
            axis=1,
        )
        diag = (
            alt.Chart(pd.DataFrame({"x": [0, 100], "y": [0, 100]}))
            .mark_line(color=PALETTE["rule"], strokeDash=[4, 4])
            .encode(x="x:Q", y="y:Q")
        )
        cbase = alt.Chart(cal).encode(
            x=alt.X(
                "predicha:Q",
                scale=alt.Scale(domain=[0, 100]),
                axis=axis("Probabilidad predicha (%)"),
            ),
        )
        cerr = cbase.mark_rule(color=PALETTE["muted"]).encode(
            y=alt.Y(
                "ic_inf:Q", scale=alt.Scale(domain=[0, 100]), axis=axis("Mortalidad observada (%)")
            ),
            y2="ic_sup:Q",
        )
        cpts = cbase.mark_point(filled=True, size=70, color=PALETTE["accent"], opacity=1).encode(
            y="observada:Q", tooltip=[alt.Tooltip("tip:N", title="Decil")]
        )
        me_rule = (
            alt.Chart(pd.DataFrame({"x": [100 * p]}))
            .mark_rule(color=PALETTE["ink"], strokeWidth=1.5)
            .encode(x="x:Q")
        )
        st.altair_chart(
            (diag + cerr + cpts + me_rule).properties(height=260), width="stretch", theme=None
        )
        mine = cal[(cal["desde"] <= p) & (cal["hasta"] >= p)]
        if mine.empty:
            mine = cal.iloc[[int(np.argmin(np.abs(cal["predicha"] - 100 * p)))]]
        m = mine.iloc[0]
        ui.caption(
            "Deciles de riesgo predicho en el conjunto de test, con intervalos de Wilson. La "
            "línea vertical marca a este paciente. Entre los pacientes de test con un riesgo "
            f"predicho similar ({pct(m['desde'], 0)} a {pct(m['hasta'], 0)}) murió el "
            f"{num(m['observada'], 0)} % (IC 95 %: {num(m['ic_inf'], 0)} a "
            f"{num(m['ic_sup'], 0)} %). "
            "Los puntos cerca de la diagonal indican que la probabilidad está bien calibrada."
        )

        # --- Puente hacia la pestaña causal ---------------------------------------------
        flip = dict(profile, rhc=1 - profile["rhc"])
        p_flip = sv.predict(art, flip)["p"]
        delta_pred = (p if profile["rhc"] else p_flip) - (p_flip if profile["rhc"] else p)
        age_c = float(profile["age"] or data["age"].median())
        aps_c = float(profile["aps1"] or data["aps1"].median())
        cate = sv.cate_at(causal, profile["dx"], age_c, aps_c)
        ui.callout(
            f"<strong>Predicción no es decisión.</strong> Si se cambia el interruptor de RHC, la "
            f"probabilidad predicha se mueve {pp(delta_pred)}. Esa diferencia refleja una "
            "asociación aprendida por el modelo predictivo, que no fue diseñado para aislar el "
            "efecto del tratamiento. La estimación causal para un perfil como este (diagnóstico, "
            f"edad y APACHE) es de {pp(cate['cate'])} (IC 95 %: {pp(cate['ic_inf'])} a "
            f"{pp(cate['ic_sup'])}); el detalle está en la pestaña Simulación causal."
        )

# ====================================================================================
# 3. Simulación causal ("What-If")
# ====================================================================================
with tab_causal:
    ui.section(
        "Qué pasaría con la mortalidad si se aplica RHC",
        "Esta pestaña responde una pregunta distinta de la predictiva: no cuál paciente tiene más "
        "riesgo, sino cuánto cambia ese riesgo por la decisión de cateterizar, P(Y | do(T)). Las "
        "estimaciones provienen de DoWhy y EconML (Fase 4 del notebook).",
    )
    bal = art.balance.copy()
    n_before = int((bal["antes"].abs() > 0.1).sum())
    n_after = int((bal["despues"].abs() > 0.1).sum())
    e1, e2 = st.columns([1.5, 1], gap="large")
    with e1:
        st.markdown("#### Balance de los confusores al ponderar por propensión")
        bal["nombre"] = bal["variable"].map(label)
        bal = bal.reindex(bal["antes"].abs().sort_values(ascending=False).index).head(15)
        order = bal["nombre"].tolist()
        long = pd.concat(
            [
                bal.assign(momento="Muestra observada", smd=bal["antes"].abs()),
                bal.assign(momento="Muestra ponderada", smd=bal["despues"].abs()),
            ]
        )
        long["tip"] = long["smd"].map(lambda v: num(v, 3))
        y = alt.Y("nombre:N", sort=order, title=None, axis=alt.Axis(labelLimit=300))
        seg = (
            alt.Chart(bal.assign(a=bal["antes"].abs(), d=bal["despues"].abs()))
            .mark_rule(color=PALETTE["rule"], strokeWidth=2.5)
            .encode(y=y, x="d:Q", x2="a:Q")
        )
        pts = (
            alt.Chart(long)
            .mark_point(size=90, opacity=1, strokeWidth=2)
            .encode(
                y=y,
                x=alt.X(
                    "smd:Q",
                    scale=alt.Scale(domain=[0, 0.55]),
                    axis=axis("Diferencia de medias estandarizada (valor absoluto)", tickCount=6),
                ),
                color=alt.Color(
                    "momento:N",
                    scale=alt.Scale(
                        domain=["Muestra observada", "Muestra ponderada"],
                        range=[PALETTE["faint"], PALETTE["accent"]],
                    ),
                    legend=alt.Legend(title=None, symbolStrokeWidth=2),
                ),
                fill=alt.Fill(
                    "momento:N",
                    scale=alt.Scale(
                        domain=["Muestra observada", "Muestra ponderada"],
                        range=["white", PALETTE["accent"]],
                    ),
                    legend=None,
                ),
                tooltip=[
                    alt.Tooltip("nombre:N", title="Variable"),
                    alt.Tooltip("momento:N", title="Muestra"),
                    alt.Tooltip("tip:N", title="Diferencia"),
                ],
            )
        )
        thr = (
            alt.Chart(pd.DataFrame({"x": [0.1]}))
            .mark_rule(color=PALETTE["muted"], strokeDash=[4, 4])
            .encode(x="x:Q")
        )
        st.altair_chart(
            (thr + seg + pts).properties(height=30 * len(bal) + 50), width="stretch", theme=None
        )
        ui.caption(
            f"Las quince covariables más desbalanceadas antes de ponderar. La línea "
            f"punteada marca el umbral de 0,1. Sin ponderar, {n_before} de "
            f"{len(art.balance)} covariables lo superan; con los pesos de propensión, "
            f"{n_after}. La ponderación construye dos grupos comparables en todo lo "
            "observado; no puede hacer lo mismo con lo que no se midió."
        )
    with e2:
        bias = NAIVE["valor"] - LR["valor"]
        st.markdown("#### Efecto promedio en la población")

        def ci(r: dict) -> str:
            return f"{pp(r['valor'])} ({num(100 * r['ic_inf'])} a {num(100 * r['ic_sup'])})"

        ui.facts(
            [
                ("Diferencia cruda", ci(NAIVE)),
                ("Regresión con ajuste", ci(LR)),
                ("Ponderación por propensión", ci(IPW)),
                ("Double Machine Learning", ci(DML)),
                (
                    "Sesgo de confusión removido",
                    f"{pp(bias)} ({num(100 * bias / NAIVE['valor'], 0)} % de la diferencia cruda)",
                ),
            ]
        )
        ui.caption(
            "Entre paréntesis, el intervalo de confianza del 95 %. Los tres estimadores "
            "ajustados coinciden en el signo y sus intervalos excluyen el cero. Double "
            "Machine Learning, que modela la gravedad de forma no lineal, da el valor más "
            "bajo. Ninguno sugiere un beneficio."
        )
        st.markdown("#### Superposición de los puntajes de propensión")
        ps = art.propensity
        edges = np.linspace(0, 1, 26)
        rows = []
        for t, name in [(0, "Sin RHC"), (1, "Con RHC")]:
            counts, _ = np.histogram(ps.loc[ps["rhc"] == t, "ps"], bins=edges)
            share = 100 * counts / counts.sum()
            rows += [
                {"x": a, "pct": s, "grupo": name}
                for a, s in zip(edges, np.r_[share, share[-1]], strict=True)
            ]
        pdf = pd.DataFrame(rows)
        pbase = alt.Chart(pdf).encode(
            x=alt.X("x:Q", axis=axis("Probabilidad estimada de recibir RHC", ".1f")),
            y=alt.Y("pct:Q", axis=axis("Pacientes (%)"), stack=None),
            color=alt.Color(
                "grupo:N",
                scale=alt.Scale(domain=["Sin RHC", "Con RHC"], range=[NO_RHC, RHC]),
                legend=alt.Legend(title=None, symbolOpacity=1, symbolType="square"),
            ),
        )
        pchart = pbase.mark_area(interpolate="step-after", opacity=0.14) + pbase.mark_line(
            interpolate="step-after", strokeWidth=2
        )
        st.altair_chart(pchart.properties(height=190), width="stretch", theme=None)
        ui.caption(
            "Los dos grupos comparten casi todo el rango de probabilidades, condición "
            "necesaria para comparar tratados y no tratados con perfiles similares "
            "(positividad). En los extremos la superposición es escasa: por eso el "
            f"notebook repite el análisis sin el {pct(CHK['recorte_proporcion'])} de pacientes "
            f"con puntajes extremos, y el efecto se mantiene en {pp(CHK['recorte_ate'])}."
        )

    # --- DAG interactivo ----------------------------------------------------------------
    st.divider()
    ui.section(
        "Supuestos causales",
        "El grafo dirigido acíclico declara qué causa qué. Cada vista destaca un conjunto de "
        "caminos y muestra qué ocurre al ajustar, o no, por cada variable.",
    )
    dview = (
        st.segmented_control(
            "Vista del grafo", list(sv.DAG_VIEWS), default="Estructura completa", key="f_dag"
        )
        or "Estructura completa"
    )
    dot = sv.dag_dot(art.dag, dview, {**PALETTE, "highlight": RHC})
    st.graphviz_chart(dot, width="stretch")
    legend = (
        "Los arcos resaltados en naranja son los caminos de esta vista."
        if sv.DAG_VIEWS[dview]["edges"]
        else ""
    )
    if sv.DAG_VIEWS[dview]["conditioned"]:
        legend += " El nodo con fondo celeste es la variable por la que se ajusta."
    ui.callout(f"<strong>{dview}.</strong> {sv.dag_text(dview, CHK)} {legend.strip()}")

    # --- Heterogeneidad -----------------------------------------------------------------
    st.divider()
    ui.section(
        "Efecto según el perfil del paciente",
        "Efecto condicional promedio (CATE) estimado con LinearDML de EconML. Por defecto se "
        "carga el perfil de la pestaña Predicción del riesgo.",
    )
    h1, h2, h3 = st.columns([1.4, 1, 1], gap="medium")
    with h1:
        dx_c = st.selectbox("Diagnóstico principal", cfg.DX_ORDER, key="c_dx")
    with h2:
        age_c = st.slider(
            "Edad (años)", 40, 85, key="c_age", help="La grilla de EconML cubre de 40 a 85 años."
        )
    with h3:
        aps_c = st.slider("Score APACHE III", 15, 140, key="c_aps")

    curve = sv.cate_curve(causal, dx_c, age_c)
    here = sv.cate_at(causal, dx_c, age_c, aps_c)
    cv = curve.assign(cate=100 * curve["cate"], lo=100 * curve["ic_inf"], hi=100 * curve["ic_sup"])
    cv["tip"] = cv.apply(
        lambda r: f"{num(r['cate'])} pp (IC 95 %: {num(r['lo'])} a {num(r['hi'])})", axis=1
    )
    k1, k2 = st.columns([1.7, 1], gap="large")
    with k1:
        cb = alt.Chart(cv).encode(
            x=alt.X("aps1:Q", axis=axis("Score APACHE III"), scale=alt.Scale(domain=[15, 140]))
        )
        band = cb.mark_area(color=PALETTE["fill"], opacity=0.85).encode(
            y=alt.Y("lo:Q", axis=axis("Cambio en la mortalidad a 30 días (pp)")), y2="hi:Q"
        )
        line = cb.mark_line(color=PALETTE["accent"], strokeWidth=2.5).encode(
            y="cate:Q",
            tooltip=[alt.Tooltip("aps1:Q", title="APACHE III"), alt.Tooltip("tip:N", title="CATE")],
        )
        zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=PALETTE["ink"]).encode(y="y:Q")
        ate_line = alt.Chart(
            pd.DataFrame({"y": [100 * DML["valor"]], "t": ["promedio poblacional"]})
        ).encode(y="y:Q")
        ate_rule = ate_line.mark_rule(color=PALETTE["muted"], strokeDash=[5, 4])
        ate_text = ate_line.mark_text(
            align="right", dy=-7, x="width", fontSize=12, fontStyle="italic", color=PALETTE["muted"]
        ).encode(text="t:N")
        dot_df = pd.DataFrame({"aps1": [aps_c], "cate": [100 * here["cate"]]})
        me_pt = (
            alt.Chart(dot_df)
            .mark_point(
                filled=True,
                size=160,
                color=PALETTE["ink"],
                stroke="white",
                strokeWidth=2,
                opacity=1,
            )
            .encode(x="aps1:Q", y="cate:Q")
        )
        st.altair_chart(
            (band + zero + ate_rule + ate_text + line + me_pt).properties(height=330),
            width="stretch",
            theme=None,
        )
        ui.caption(
            "Banda: intervalo de confianza del 95 %. Línea punteada: efecto promedio de "
            f"Double ML. La curva se obtiene de una grilla de edades ({GRID_AGES} años) "
            "interpolada linealmente. La pendiente respecto del APACHE no es "
            "estadísticamente significativa: es un indicio, no un hallazgo."
        )
    with k2:
        state = {
            "dañino": ("Aumento significativo de la mortalidad", PALETTE["bad"]),
            "beneficioso": ("Reducción significativa de la mortalidad", PALETTE["good"]),
            "sin efecto demostrable": ("Efecto no distinguible de cero", PALETTE["warn"]),
        }
        title, color = state[here["lectura"]]
        st.markdown(f"#### Paciente de {age_c} años, APACHE {aps_c}")
        ui.signal(
            dx_c,
            title,
            color,
            f"Aplicar RHC cambia la probabilidad de morir a los 30 días en "
            f"<strong>{pp(here['cate'])}</strong>.",
            f"IC 95 %: {pp(here['ic_inf'])} a {pp(here['ic_sup'])}",
        )
        if here["lectura"] == "dañino":
            nnh = 1 / here["cate"]
            ui.caption(
                f"Equivale a una muerte adicional cada {num(nnh, 0)} pacientes cateterizados "
                "con este perfil (número necesario para dañar)."
            )
        ui.caption(
            "Al recorrer muchos perfiles, algunos intervalos excluyen el cero por azar. La "
            "evidencia firme de heterogeneidad se limita al subgrupo con sepsis; el resto "
            "de las diferencias entre perfiles debe leerse como exploratoria."
        )
        st.space("small")
        st.markdown("#### Impacto de una política")
        n_pol = st.slider(
            "Pacientes con este perfil cateterizados por año", 100, 5000, 1000, 100, key="c_n"
        )
        lo_d, hi_d = n_pol * here["ic_inf"], n_pol * here["ic_sup"]
        ui.facts(
            [
                ("Muertes adicionales esperadas", num(n_pol * here["cate"], 0)),
                ("Intervalo de confianza del 95 %", f"{num(lo_d, 0)} a {num(hi_d, 0)}"),
            ]
        )

    sub = pd.DataFrame(causal["subgrupos"])
    sub = sub.assign(
        cate=100 * sub["CATE (LinearDML)"],
        lo=100 * sub["IC_inf"],
        hi=100 * sub["IC_sup"],
        bosque=100 * sub["CATE (bosque causal)"],
    )
    sub["etiqueta"] = sub.apply(lambda r: f"{r['diagnóstico']} (n = {num(r['n'], 0)})", axis=1)
    sub["tip"] = sub.apply(
        lambda r: (
            f"{num(r['cate'])} pp (IC 95 %: {num(r['lo'])} a "
            f"{num(r['hi'])}); bosque causal {num(r['bosque'])} pp"
        ),
        axis=1,
    )
    order = sub.sort_values("cate", ascending=False)["etiqueta"].tolist()
    st.markdown("#### Efecto por diagnóstico principal")
    sb = alt.Chart(sub).encode(
        y=alt.Y("etiqueta:N", sort=order, title=None, axis=alt.Axis(labelLimit=320))
    )
    s_rule = sb.mark_rule(strokeWidth=2.5, strokeCap="round", color=PALETTE["ink"]).encode(
        x=alt.X("lo:Q", axis=axis("Cambio en la mortalidad a 30 días (pp)")),
        x2="hi:Q",
        tooltip=[alt.Tooltip("tip:N", title="Efecto")],
    )
    s_dot = sb.mark_point(filled=True, size=110, color=PALETTE["ink"], opacity=1).encode(
        x="cate:Q", tooltip=[alt.Tooltip("tip:N", title="Efecto")]
    )
    s_forest = sb.mark_point(
        shape="diamond", size=90, color=PALETTE["accent"], filled=False, strokeWidth=2
    ).encode(x="bosque:Q", tooltip=[alt.Tooltip("tip:N", title="Efecto")])
    s_zero = (
        alt.Chart(pd.DataFrame({"x": [0]}))
        .mark_rule(color=PALETTE["muted"], strokeDash=[3, 3])
        .encode(x="x:Q")
    )
    st.altair_chart(
        (s_zero + s_rule + s_dot + s_forest).properties(height=38 * len(sub) + 40),
        width="stretch",
        theme=None,
    )
    ui.caption(
        "Círculo y línea: LinearDML con su intervalo del 95 %. Rombo: bosque causal "
        "(CausalForestDML). Solo en el subgrupo con falla multiorgánica y sepsis el aumento "
        "de la mortalidad es estadísticamente distinto de cero. Ningún diagnóstico muestra "
        "un beneficio significativo."
    )

    # --- Refutaciones ------------------------------------------------------------------
    st.divider()
    ui.section(
        "Semáforo de robustez",
        "Tres pruebas de refutación de DoWhy, con 100 simulaciones cada una, sobre el efecto "
        f"estimado por regresión ({pp(LR['valor'])}). Un estimador confiable debe resistir "
        "las tres.",
    )
    descriptions = {
        "Causa común aleatoria": "Agrega un confusor aleatorio. Si el modelo es correcto, el "
        "efecto no debería cambiar.",
        "Tratamiento placebo": "Reemplaza el tratamiento por una permutación al azar. El efecto "
        "debería desaparecer.",
        "Subconjunto de datos (80 %)": "Reestima el efecto en submuestras del 80 %. El efecto "
        "debería mantenerse dentro del intervalo original.",
    }
    r_cols = st.columns(3, gap="large", border=True)
    for col, (name, r) in zip(r_cols, causal["refutaciones"].items(), strict=True):
        with col:
            ok = r["supera"]
            ui.signal(
                name,
                "Superada" if ok else "No superada",
                PALETTE["good"] if ok else PALETTE["bad"],
                descriptions.get(name, r["criterio"]),
                f"Efecto refutado: {pp(r['ate_refutado'], 2)}. p = {num(r['p_valor'], 2)}. "
                f"Criterio: {r['criterio']}.",
            )
    ev_ = causal["e_value"]
    with st.container(border=True):
        ui.signal(
            "Confusión no observada",
            "Advertencia",
            PALETTE["warn"],
            f"Ninguna prueba descarta un confusor no medido, como el criterio del médico o el "
            f"hospital. El E-value es {num(ev_['puntual'], 2)}: para anular el efecto, ese "
            f"confusor debería asociarse con el RHC y con la muerte con un riesgo relativo de al "
            f"menos {num(ev_['puntual'], 2)} en ambos casos. Con uno de "
            f"{num(ev_['limite_inferior'], 2)} bastaría para que el intervalo de confianza "
            "incluya el cero.",
            "Valores moderados: la conclusión es razonable con los datos disponibles, pero no "
            "reemplaza a un ensayo aleatorizado.",
        )

# ====================================================================================
# 4. Sobre el trabajo
# ====================================================================================
with tab_about:
    ui.section("Sobre el trabajo")
    w1, w2 = st.columns([1.5, 1], gap="large")
    with w1:
        st.markdown(
            f"""
#### Pregunta

El cateterismo cardíaco derecho mide de forma directa la función cardíaca de pacientes críticos.
Su uso se extendió sin evidencia experimental de beneficio. El trabajo estima el riesgo de muerte a
30 días de cada paciente (pregunta predictiva) y el cambio en ese riesgo atribuible a la decisión de
cateterizar (pregunta causal).

#### Datos

Estudio SUPPORT, analizado por Connors et al. (1996): {N_PAC} pacientes adultos internados en
terapia intensiva en cinco hospitales universitarios de Estados Unidos entre 1989 y 1994, de los
cuales {N_RHC} recibieron RHC en las primeras 24 horas. La auditoría forense detectó
{N_SENT} valores centinela, faltantes no aleatorios y una fuerte confusión por indicación.

#### Método

1. **Inferencia clásica.** Auditoría de datos, contrastes de hipótesis con verificación de
   supuestos y ANOVA por diagnóstico.
2. **Regresión lineal.** Mínimos cuadrados para la duración de la internación, diagnóstico de
   Gauss-Markov, puntos influyentes, multicolinealidad y PCA sin fuga de datos.
3. **Torneo predictivo.** Regresión logística, Naive Bayes, árbol de decisión y Random Forest con
   validación cruzada estratificada, calibración y análisis de umbrales.
4. **Inferencia causal.** DAG, criterio de puerta trasera, regresión ajustada, ponderación por
   propensión, Double Machine Learning, bosques causales y pruebas de refutación.

#### Límites

Los resultados dependen del supuesto de que no hay confusores relevantes sin medir. La muestra
proviene de hospitales universitarios de los años noventa, y la práctica clínica cambió desde
entonces. La aplicación no reemplaza el juicio clínico.
"""
        )
    with w2:
        st.markdown("#### Equipo")
        ui.facts(
            [
                ("Project Lead y Scrum Master", "Valentina Bolívar"),
                ("Backend, Machine Learning e inferencia causal", "Valentina Bolívar"),
                ("Frontend y experiencia de usuario", "Felicitas Paz"),
                ("QA y validación", "Emilia Bedners"),
            ]
        )
        st.space("small")
        st.markdown("#### Recursos")
        st.markdown(
            f"""
- [Repositorio en GitHub]({REPO})
- [Notebook reproducible]({REPO}/blob/main/notebooks/TP_Integrador_Final.ipynb)
- [Bitácora de ingeniería]({REPO}/blob/main/BITACORA.md)
"""
        )
        st.markdown("#### Referencias")
        st.markdown(
            """
Connors, A. F., Speroff, T., Dawson, N. V., et al. (1996). The effectiveness of right heart
catheterization in the initial care of critically ill patients. *JAMA, 276*(11), 889–897.

Chernozhukov, V., et al. (2018). Double/debiased machine learning for treatment and structural
parameters. *The Econometrics Journal, 21*(1), C1–C68.

VanderWeele, T. J., y Ding, P. (2017). Sensitivity analysis in observational research:
introducing the E-value. *Annals of Internal Medicine, 167*(4), 268–274.
"""
        )

st.markdown(
    f"""<p class="foot">Cátedra de Estadística Aplicada y Modelos de Regresión, prof. Ivan Gonzalo
Vilaboa. Los resultados provienen del notebook del repositorio; la aplicación no reestima los
modelos. Datos: Vanderbilt Biostatistics. <a href="{REPO}">Código fuente</a>.</p>""",
    unsafe_allow_html=True,
)
