"""Unit tests for configuration loading and validation."""

from pathlib import Path
import pytest
from fever_ir.config import load_config, AppConfig


def test_load_default_config() -> None:
    """Verify that configs/default.yaml loads properly into AppConfig."""
    config_path = Path("configs/default.yaml")
    assert config_path.is_file(), "configs/default.yaml must exist."

    config = load_config(config_path)
    assert isinstance(config, AppConfig)
    assert config.seed == 42
    assert config.retrieval.depth == 100
    assert config.retrieval.report_k == [1, 3, 5, 10, 20]
    assert config.sampling.tune_claims == 500
    assert config.sampling.eval_claims == 1500
    assert config.bm25.k1 == 1.5
    assert config.bm25.b == 0.75
    assert config.dense.model_name == "all-MiniLM-L6-v2"
    assert config.rrf.k == 60


def test_missing_config_raises() -> None:
    """Verify that attempting to load a non-existent config file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_config("configs/non_existent.yaml")
