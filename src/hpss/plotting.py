"""One visual style for every figure in the report.

* Colors follow the *method*, not the figure: NumPy is always blue, pure Python
  is always orange, ... so a reader learns the mapping once.
* The 8 colors are a validated categorical palette (colour-blind safe in this
  order); every series also gets its own marker shape as a second cue.
* Thin lines, light solid grid, no dual y-axes, conclusion-style titles.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from . import config

# validated categorical order: blue, orange, aqua, yellow, magenta, green, violet, red
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Fixed method -> slot mapping used across all figures. The three entities that
# appear together most often (NumPy/BLAS, pure Python, our final system) take the
# first three slots, the only ones that stay distinguishable in every pairing.
# Series order inside each figure is chosen so that neighbouring legend entries
# pass the colour-blindness check (validated with the dataviz palette validator).
METHOD_SLOT = {
    "numpy": 0, "blas": 0, "dense": 0, "exact": 0,
    "python": 1, "naive": 1, "sparse": 1,
    "final": 2, "rerank": 2, "embedding": 2,
    "numba": 3, "power_iteration": 3,
    "numba_parallel": 4, "truncated_svd": 4,
    "numba_fastmath": 5, "pca": 5, "dask": 5,
    "threadpool": 6, "numpy_svd": 6,
    "processpool": 7, "ray": 7, "lsh": 7, "asyncio": 7,
}


def color(method: str) -> str:
    return PALETTE[METHOD_SLOT.get(method, 0) % len(PALETTE)]


def marker(method: str) -> str:
    return MARKERS[METHOD_SLOT.get(method, 0) % len(MARKERS)]


def setup_style():
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "font.size": 10.5,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 10.5,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "grid.linestyle": "-",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "text.color": INK,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "lines.linewidth": 1.8,
        "lines.markersize": 6,
        "patch.linewidth": 0,
    })


def line(ax, x, y, method, label=None, **kw):
    """Plot one series with the method's fixed colour + marker (white ring on markers)."""
    kw.setdefault("linewidth", 1.8)
    kw.setdefault("markersize", 6.5)
    return ax.plot(x, y, color=color(method), marker=marker(method),
                   markeredgecolor="white", markeredgewidth=1.0,
                   label=label or method, **kw)


def label_end(ax, x, y, text, method, dx=4, dy=0):
    """Direct label at the last point of a series (selective, not on every point)."""
    ax.annotate(text, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
                color=INK_2, fontsize=9, va="center")


def bars(ax, names, values, methods, fmt="{:.0f}x", log=False, horizontal=False, labels=None):
    """Bar chart with one colour per method and a value label just outside each bar.
    `labels` overrides the formatted values (e.g. "12.3 ms (450x)")."""
    cols = [color(m) for m in methods]
    pos = np.arange(len(names))
    labels = labels or [fmt.format(v) for v in values]
    if horizontal:
        b = ax.barh(pos, values, color=cols, height=0.62, edgecolor="white", linewidth=2)
        ax.set_yticks(pos, names)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        for rect, lab in zip(b, labels):
            ax.annotate(lab, xy=(rect.get_width(), rect.get_y() + rect.get_height() / 2),
                        xytext=(4, 0), textcoords="offset points", va="center",
                        fontsize=9, color=INK_2)
        if log:
            ax.set_xscale("log")
    else:
        b = ax.bar(pos, values, color=cols, width=0.62, edgecolor="white", linewidth=2)
        ax.set_xticks(pos, names)
        ax.grid(axis="x", visible=False)
        for rect, lab in zip(b, labels):
            ax.annotate(lab, xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center",
                        fontsize=9, color=INK_2)
        if log:
            ax.set_yscale("log")
    return b


def savefig(fig, name: str, stage: str):
    """Save PNG (slides) + PDF (report, vector) into results/<stage>/."""
    fig.tight_layout()
    out = config.stage_dir(stage)
    png = out / f"{name}.png"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    print(f"saved -> results/{stage}/{name}.png (+ .pdf)")
    return png


def amdahl(n, p):
    """Amdahl's law: speedup with n workers when a fraction p of the work is parallel."""
    n = np.asarray(n, dtype=float)
    return 1.0 / ((1.0 - p) + p / n)


def fit_amdahl(workers, speedups):
    """Least-squares fit of the parallel fraction p in Amdahl's law."""
    from scipy.optimize import curve_fit
    (p,), _ = curve_fit(amdahl, np.asarray(workers, float), np.asarray(speedups, float),
                        p0=[0.9], bounds=(0.0, 1.0))
    return float(p)
