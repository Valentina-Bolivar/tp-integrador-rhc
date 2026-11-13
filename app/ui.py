"""Identidad visual de la aplicación: paleta, tipografía, estilos y tema de Altair."""

from __future__ import annotations

import math
from decimal import ROUND_HALF_UP, Decimal

import altair as alt
import streamlit as st

FONT = "Newsreader, Georgia, serif"

PALETTE = {
    "bg": "#F5F8FA",
    "surface": "#FFFFFF",
    "wash": "#EAF1F6",
    "fill": "#D6E6F2",
    "ink": "#1E2A33",
    "muted": "#3D4A55",
    "faint": "#5A6772",
    "rule": "#DCE4EA",
    "accent": "#2F6E9B",
    "highlight": "#e0703c",
    "good": "#2E8A63",
    "warn": "#B7862A",
    "bad": "#B8483F",
}

# Colores de datos (validados para daltonismo y contraste, los mismos de las figuras del notebook).
NO_RHC = "#2a78d6"
RHC = "#e0703c"
LOWER = "#2a78d6"  # reduce el riesgo, sobrevivió
HIGHER = "#e0703c"  # aumenta el riesgo, murió

CSS = f"""
<style>
:root {{
  --ink: {PALETTE["ink"]};
  --muted: {PALETTE["muted"]};
  --rule: {PALETTE["rule"]};
  --wash: {PALETTE["wash"]};
  --accent: {PALETTE["accent"]};
}}
html, body {{ font-variant-numeric: lining-nums tabular-nums; }}
.block-container {{ max-width: 1240px; padding-top: 2.2rem; padding-bottom: 4rem; }}
header[data-testid="stHeader"] {{ background: transparent; }}
#MainMenu, footer {{ visibility: hidden; }}
h1, h2, h3 {{ letter-spacing: -0.01em; color: var(--ink); }}
h2 {{ margin-top: 0.4rem; }}
p, li {{ line-height: 1.6; }}

.masthead {{ border-bottom: 1px solid var(--rule); padding-bottom: 1.4rem; margin-bottom: 0.6rem; }}
.masthead .kicker {{ color: var(--muted); font-size: 0.98rem; font-style: italic; margin: 0; }}
.masthead h1 {{ font-size: 2.9rem; font-weight: 450; line-height: 1.08; margin: 0.35rem 0 0.9rem 0;
  padding: 0; }}
.masthead .lead {{ font-size: 1.32rem; line-height: 1.5; max-width: 46rem; color: var(--ink);
  margin: 0; font-weight: 400; }}
.masthead .lead strong {{ font-weight: 600; }}
.masthead .note {{ color: var(--muted); font-size: 0.95rem; margin-top: 0.7rem; max-width: 46rem; }}

.section-intro {{ color: var(--muted); font-size: 1.05rem; max-width: 50rem;
  margin: -0.3rem 0 1rem 0; }}
.caption {{ color: var(--muted); font-size: 0.97rem; line-height: 1.55; }}

.riskcard {{ background: {PALETTE["surface"]}; border: 1px solid var(--rule); border-radius: 0.9rem;
  padding: 1.4rem 1.6rem 1.2rem 1.6rem; margin-bottom: 1.2rem; }}
.riskcard .label {{ color: var(--muted); font-size: 1rem; margin: 0; }}
.riskcard .value {{ font-size: 4rem; font-weight: 420; line-height: 1; margin: 0.4rem 0 0.2rem 0;
  color: var(--ink); }}
.riskcard .value small {{ font-size: 1.6rem; color: var(--muted); font-weight: 400; }}
.riskcard .ci {{ color: var(--muted); font-size: 1rem; margin: 0 0 1rem 0; }}
.riskcard .verdict {{ font-size: 1.08rem; margin: 0.9rem 0 0 0; }}

.riskbar {{ position: relative; height: 0.65rem; border-radius: 1rem; margin: 1.9rem 0 2.4rem 0;
  background: linear-gradient(90deg, #DCEAF5 0%, #F3E3D8 100%); }}
.riskbar .band {{ position: absolute; top: 0; height: 100%; background: rgba(30,42,51,0.18);
  border-radius: 1rem; margin: 0; }}
.riskbar .dot {{ position: absolute; top: 50%; width: 1.05rem; height: 1.05rem; border-radius: 50%;
  background: var(--ink); border: 3px solid #fff; transform: translate(-50%, -50%);
  box-shadow: 0 0 0 1px var(--ink); }}
.riskbar .thr {{ position: absolute; top: -0.55rem; height: 1.75rem; width: 0;
  border-left: 1.5px dashed var(--muted); }}
.riskbar .thr span {{ position: absolute; top: -1.35rem; left: 0; transform: translateX(-50%);
  white-space: nowrap; font-size: 0.82rem; color: var(--muted); }}
.riskbar .tick {{ position: absolute; top: 1rem; font-size: 0.86rem; color: var(--faint);
  transform: translateX(-50%); white-space: nowrap; }}
.riskbar .tick.first {{ transform: none; }}
.riskbar .tick.last {{ transform: translateX(-100%); }}

.signal {{ display: flex; gap: 0.9rem; align-items: flex-start; }}
.signal .lamp {{ flex: 0 0 auto; width: 0.95rem; height: 0.95rem; border-radius: 50%;
  margin-top: 0.38rem; }}
.signal h4 {{ margin: 0 0 0.15rem 0; font-size: 1.12rem; font-weight: 600; padding: 0; }}
.signal .state {{ font-size: 0.95rem; margin: 0 0 0.5rem 0; }}
.signal p {{ margin: 0 0 0.35rem 0; font-size: 0.97rem; }}
.signal .fig {{ color: var(--muted); font-size: 0.93rem; }}

.callout {{ border-left: 3px solid var(--accent); background: var(--wash); padding: 0.9rem 1.15rem;
  border-radius: 0 0.6rem 0.6rem 0; margin: 0.6rem 0 1rem 0; }}
.callout p {{ margin: 0; }}

.factrow {{ display: grid; grid-template-columns: 1fr auto; gap: 0.2rem 1rem; padding: 0.45rem 0;
  border-bottom: 1px solid var(--rule); font-size: 1rem; }}
.factrow span:last-child {{ text-align: right; color: var(--ink); }}
.factrow span:first-child {{ color: var(--muted); }}

div[data-testid="stMetric"] {{ background: {PALETTE["surface"]}; }}
div[data-testid="stMetricLabel"] p {{ color: var(--muted); font-size: 0.98rem; }}
div[data-testid="stMetricValue"] {{ font-weight: 450; }}

button[data-baseweb="tab"] p {{ font-size: 1.12rem; }}
div[data-baseweb="tab-list"] {{ gap: 1.6rem; border-bottom: 1px solid var(--rule); }}

.foot {{ color: var(--muted); font-size: 0.95rem; border-top: 1px solid var(--rule);
  margin-top: 3rem; padding-top: 1rem; }}
@media (max-width: 640px) {{
  .masthead h1 {{ font-size: 2.1rem; }}
  .masthead .lead {{ font-size: 1.12rem; }}
  .riskcard .value {{ font-size: 3rem; }}
}}
</style>
"""


def inject() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


@alt.theme.register("rhc", enable=True)
def altair_theme() -> alt.theme.ThemeConfig:
    ink, muted, rule = PALETTE["ink"], PALETTE["muted"], PALETTE["rule"]
    return {
        "config": {
            "font": FONT,
            "background": "transparent",
            "view": {"stroke": None},
            "axis": {
                "labelFont": FONT,
                "titleFont": FONT,
                "labelFontSize": 13,
                "titleFontSize": 14,
                "titleFontWeight": 500,
                "labelColor": muted,
                "titleColor": ink,
                "gridColor": rule,
                "gridOpacity": 0.7,
                "domainColor": rule,
                "tickColor": rule,
                "titlePadding": 10,
            },
            "axisY": {"domain": False, "ticks": False},
            "legend": {
                "labelFont": FONT,
                "titleFont": FONT,
                "labelFontSize": 13,
                "titleFontSize": 13,
                "titleFontWeight": 500,
                "labelColor": ink,
                "titleColor": muted,
                "orient": "top",
                "symbolStrokeWidth": 0,
            },
            "header": {"labelFont": FONT, "titleFont": FONT, "labelFontSize": 13},
            "text": {"font": FONT, "fontSize": 13, "color": ink},
            "title": {
                "font": FONT,
                "fontSize": 15,
                "fontWeight": 500,
                "color": ink,
                "anchor": "start",
            },
            "range": {"category": [NO_RHC, RHC, "#1f9e74", "#b27a14"]},
        }
    }


def num(x: float, decimals: int = 1) -> str:
    """Número con coma decimal y punto de miles, como se escribe en español.

    Redondea hacia arriba en el empate a partir de la representación decimal del número, para
    que un valor guardado como 0,1765 se muestre como 0,177, igual que en el notebook.
    """
    if not math.isfinite(x):
        return "sin dato"
    q = Decimal(1).scaleb(-decimals)
    s = f"{Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP):,.{decimals}f}"
    return s.replace(",", "_").replace(".", ",").replace("_", ".")


def pct(x: float, decimals: int = 1) -> str:
    return f"{num(100 * x, decimals)}\u00a0%"


def pp(x: float, decimals: int = 1, sign: bool = True) -> str:
    """Puntos porcentuales a partir de una diferencia de proporciones."""
    value = num(100 * x, decimals)
    if sign and x > 0:
        value = "+" + value
    return f"{value}\u00a0pp"


def section(title: str, intro: str | None = None) -> None:
    st.markdown(f"## {title}")
    if intro:
        st.markdown(f'<p class="section-intro">{intro}</p>', unsafe_allow_html=True)


def caption(text: str) -> None:
    st.markdown(f'<p class="caption">{text}</p>', unsafe_allow_html=True)


def callout(text: str) -> None:
    st.markdown(f'<div class="callout"><p>{text}</p></div>', unsafe_allow_html=True)


def facts(rows: list[tuple[str, str]]) -> None:
    html = "".join(f'<div class="factrow"><span>{a}</span><span>{b}</span></div>' for a, b in rows)
    st.markdown(html, unsafe_allow_html=True)


def signal(title: str, state: str, color: str, body: str, figure: str) -> None:
    st.markdown(
        f"""<div class="signal"><div class="lamp" style="background:{color}"></div><div>
        <h4>{title}</h4><p class="state" style="color:{color}">{state}</p>
        <p>{body}</p><p class="fig">{figure}</p></div></div>""",
        unsafe_allow_html=True,
    )
