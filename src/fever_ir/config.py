"""Configuration schema and loader for FEVERScope.

Follows Single Responsibility Principle (SRP):
Exclusively responsible for parsing, validating, and providing typed access
to system configuration defined in YAML files.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml


@dataclass(frozen=True)
class PathsConfig:
    raw: Path
    interim: Path
    processed: Path
    indexes: Path
    results: Path


@dataclass(frozen=True)
class MiniSamplingConfig:
    claims_per_label: int = 30
    nei_claims: int = 10
    random_pages: int = 400


@dataclass(frozen=True)
class SamplingConfig:
    tune_claims: int = 500
    eval_claims: int = 1500
    nei_claims: int = 200
    distractor_train_claims: int = 5000
    target_corpus_sentences: int = 200000
    mini: MiniSamplingConfig = field(default_factory=MiniSamplingConfig)


@dataclass(frozen=True)
class PreprocessingConfig:
    use_title_augmentation: bool = True
    lowercase: bool = True
    remove_stopwords: bool = True
    stemming: bool = False


@dataclass(frozen=True)
class RetrievalConfig:
    depth: int = 100
    report_k: list[int] = field(default_factory=lambda: [1, 3, 5, 10, 20])


@dataclass(frozen=True)
class BM25Config:
    k1: float = 1.5
    b: float = 0.75


@dataclass(frozen=True)
class DenseConfig:
    model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = 64
    normalize: bool = True


@dataclass(frozen=True)
class RRFConfig:
    k: int = 60


@dataclass(frozen=True)
class AppConfig:
    seed: int
    paths: PathsConfig
    sampling: SamplingConfig
    preprocessing: PreprocessingConfig
    retrieval: RetrievalConfig
    bm25: BM25Config
    dense: DenseConfig
    rrf: RRFConfig

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AppConfig":
        paths_data = data.get("paths", {})
        paths = PathsConfig(
            raw=Path(paths_data.get("raw", "data/raw")),
            interim=Path(paths_data.get("interim", "data/interim")),
            processed=Path(paths_data.get("processed", "data/processed")),
            indexes=Path(paths_data.get("indexes", "indexes")),
            results=Path(paths_data.get("results", "results")),
        )

        sampling_data = data.get("sampling", {})
        mini_data = sampling_data.get("mini", {})
        mini_sampling = MiniSamplingConfig(
            claims_per_label=mini_data.get("claims_per_label", 30),
            nei_claims=mini_data.get("nei_claims", 10),
            random_pages=mini_data.get("random_pages", 400),
        )
        sampling = SamplingConfig(
            tune_claims=sampling_data.get("tune_claims", 500),
            eval_claims=sampling_data.get("eval_claims", 1500),
            nei_claims=sampling_data.get("nei_claims", 200),
            distractor_train_claims=sampling_data.get("distractor_train_claims", 5000),
            target_corpus_sentences=sampling_data.get("target_corpus_sentences", 200000),
            mini=mini_sampling,
        )

        preproc_data = data.get("preprocessing", {})
        preprocessing = PreprocessingConfig(
            use_title_augmentation=preproc_data.get("use_title_augmentation", True),
            lowercase=preproc_data.get("lowercase", True),
            remove_stopwords=preproc_data.get("remove_stopwords", True),
            stemming=preproc_data.get("stemming", False),
        )

        retrieval_data = data.get("retrieval", {})
        retrieval = RetrievalConfig(
            depth=retrieval_data.get("depth", 100),
            report_k=retrieval_data.get("report_k", [1, 3, 5, 10, 20]),
        )

        bm25_data = data.get("bm25", {})
        bm25 = BM25Config(
            k1=float(bm25_data.get("k1", 1.5)),
            b=float(bm25_data.get("b", 0.75)),
        )

        dense_data = data.get("dense", {})
        dense = DenseConfig(
            model_name=dense_data.get("model_name", "all-MiniLM-L6-v2"),
            batch_size=dense_data.get("batch_size", 64),
            normalize=dense_data.get("normalize", True),
        )

        rrf_data = data.get("rrf", {})
        rrf = RRFConfig(
            k=int(rrf_data.get("k", 60)),
        )

        return cls(
            seed=data.get("seed", 42),
            paths=paths,
            sampling=sampling,
            preprocessing=preprocessing,
            retrieval=retrieval,
            bm25=bm25,
            dense=dense,
            rrf=rrf,
        )


def load_config(config_path: Path | str = "configs/default.yaml") -> AppConfig:
    """Load configuration from a YAML file."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return AppConfig.from_dict(data)
