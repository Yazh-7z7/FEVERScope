"""Visualization tools for Precision-Recall curves and method comparisons.

Follows Single Responsibility Principle (SRP):
Responsible solely for generating and saving matplotlib evaluation figures.
"""

from pathlib import Path
import matplotlib.pyplot as plt


def plot_precision_recall_curves(
    pr_data: dict[str, tuple[list[float], list[float]]],
    output_path: Path | str,
    title: str = "Precision-Recall Curve (K=1..100)",
) -> None:
    """Plot macro-averaged Precision-Recall curves across methods.

    Args:
        pr_data: Dict mapping method_name -> (recalls, precisions).
        output_path: Target path to save the generated figure.
        title: Chart title.
    """
    plt.figure(figsize=(8, 6))

    for method, (recalls, precisions) in pr_data.items():
        plt.plot(recalls, precisions, marker="o", markersize=3, label=method)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300)
    plt.close()
