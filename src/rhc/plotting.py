"""Estilo gráfico común: paleta validada para daltonismo, tipografía serif, ejes sobrios."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Paleta categórica validada (contraste ≥ 3:1 y separación CVD sobre fondo claro).
NO_RHC = "#2a78d6"
RHC = "#e0703c"
THIRD = "#1f9e74"
INK = "#1d1d1b"
MUTED = "#6b6a66"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
GROUP_COLORS = {0: NO_RHC, 1: RHC, "Sin RHC": NO_RHC, "Con RHC": RHC}

# Divergente: dos polos + punto medio gris neutro (para correlaciones).
DIVERGING = LinearSegmentedColormap.from_list("rhc_div", [NO_RHC, "#f1f0ec", RHC])
# Secuencial de un solo tono (celeste → azul profundo).
SEQUENTIAL = LinearSegmentedColormap.from_list("rhc_seq", ["#eef4fa", "#8fb3dc", "#1f4f8a"])


def set_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Liberation Serif", "Times New Roman"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelcolor": INK,
            "axes.edgecolor": MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "legend.frameon": False,
            "lines.linewidth": 2,
        }
    )


def figure_title(fig: plt.Figure, number: str, text: str) -> None:
    """Título numerado de figura, alineado a la izquierda."""
    fig.suptitle(f"Figura {number}. {text}", x=0.01, ha="left", fontsize=12, fontweight="bold")
