"""Shared Matplotlib settings for SL_GPsim figures (sizes in points)."""

from cycler import cycler

INK = "#202020"
BLUE = "#0072B2"
ORANGE = "#C55305"
TEAL = "#007E69"
PURPLE = "#81579A"
COLUMN_WIDTH = 183 / 25.4  # Double-column width in inches.

FIGURE_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 8,
    "axes.titleweight": "normal",
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.grid": False,
    "axes.prop_cycle": cycler(color=[BLUE, ORANGE, TEAL, PURPLE, INK]),
    "lines.linewidth": 1,
    "legend.frameon": False,
    "legend.fontsize": 6.5,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 500,
    "svg.fonttype": "none",  # Keep vector labels editable.
    "svg.hashsalt": "SL_GPsim",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}
