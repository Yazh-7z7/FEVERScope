"""Evaluation suite: Information retrieval metrics, runner, statistical tests, and plots."""

from fever_ir.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    average_precision,
    ndcg_at_k,
    reciprocal_rank,
    set_coverage_at_k,
)
from fever_ir.evaluation.runner import evaluate_run
from fever_ir.evaluation.significance import paired_bootstrap_ci, paired_wilcoxon_test
from fever_ir.evaluation.plots import plot_precision_recall_curves

__all__ = [
    "precision_at_k",
    "recall_at_k",
    "average_precision",
    "ndcg_at_k",
    "reciprocal_rank",
    "set_coverage_at_k",
    "evaluate_run",
    "paired_bootstrap_ci",
    "paired_wilcoxon_test",
    "plot_precision_recall_curves",
]
