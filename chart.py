"""Shared plotting helper for monitor.py and serve_kv.py."""
import itertools
import sys


def save_plot(t, series, hlines, path, title="Memory usage", ylabel="GB", xlabel="seconds"):
    """series: dict[label -> list[float]] plotted as lines against t.
    hlines: dict[label -> value] plotted as dashed horizontal reference lines.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("--plot requires matplotlib: pip install matplotlib")

    fig, ax = plt.subplots(figsize=(10, 5))
    for label, values in series.items():
        ax.plot(t, values, label=label)
    colors = itertools.cycle(["red", "orange", "purple", "brown"])
    for (label, value), color in zip(hlines.items(), colors):
        ax.axhline(value, color=color, linestyle="--", linewidth=1, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
