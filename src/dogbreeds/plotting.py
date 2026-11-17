"""Style et fonctions communes pour les figures."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .data import FIGURES_DIR  # noqa: E402

INK = "#1f2933"
BLUE = "#1f6fb4"
RED = "#c00000"
ORANGE = "#eb6834"
LIGHT_BLUE = "#2a78d6"
GREY = "#cfd3d8"


def clean_axes(ax, grid=None):
    """Axes discrets, sans bordure haut/droite. grid : None, "y" ou "both" (pointillés)."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d1d5db")
    ax.tick_params(colors=INK, labelsize=9)
    if grid:
        ax.grid(True, axis="both" if grid == "both" else "y", linestyle="--", color="#b8bcc2", linewidth=0.8)
        ax.set_axisbelow(True)


def save_figure(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / name, bbox_inches="tight")
    plt.close(fig)
    print(f"saved results/figures/{name}")
