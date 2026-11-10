from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path


def _default_database_url() -> str:
    return os.environ.get("DATABASE_URL", "postgresql://rag:rag@localhost:5433/rag")


@dataclass(frozen=True)
class Settings:
    database_url: str = field(default_factory=_default_database_url)
    embed_dim: int = 1024
    chunk_target_tokens: int = 350
    chunk_max_tokens: int = 500
    curated_max_tokens: int = 1500
    candidates_per_leg: int = 50
    rrf_k: int = 60
    rerank_n: int = 40
    hnsw_ef_search: int = 200            # raised to candidates_per_leg if lower
    lexical_common_fraction: float = 0.02
    lexical_common_floor: int = 1000
    max_parents: int = 8
    parent_cap_tokens: int = 1200
    evidence_budget_tokens: int = 6000
    gate_retries: int = 1
    max_delete_fraction: float = 0.2
    delete_guard_floor: int = 10
    tau: float = 0.30  # provisional; replaced from the committed calibration file


def calibrated(settings: Settings, path: Path, reranker_id: str) -> Settings:
    """Return settings with tau taken from the calibration file, if one exists.

    Raises ValueError if the file was fitted to a different reranker or holds
    a tau outside (0, 1). The provenance keys are not read.
    """
    if not path.exists():
        return settings
    data = json.loads(path.read_text())
    if data.get("reranker") != reranker_id:
        raise ValueError(f"calibration fitted to {data.get('reranker')!r}, not {reranker_id!r}")
    tau = data.get("tau")
    if not isinstance(tau, (int, float)) or isinstance(tau, bool) or not 0.0 < tau < 1.0:
        raise ValueError(f"calibrated tau must be strictly between 0 and 1, got {tau!r}")
    return replace(settings, tau=float(tau))
