"""Environment-driven configuration (prefix ``SIGNALSIFT_``)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SIGNALSIFT_", env_file=".env", extra="ignore")

    # Embeddings
    embedder: Literal["lsa", "sbert"] = "lsa"
    sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    lsa_dims: int = Field(128, ge=8, le=1024)

    # Retrieval
    vector_backend: Literal["numpy", "faiss"] = "numpy"
    reranker: Literal["none", "heuristic", "cross-encoder"] = "heuristic"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_candidates: int = Field(50, ge=10, le=500)

    # Topics
    cluster_method: Literal["kmeans", "bertopic"] = "kmeans"
    n_topics: int = Field(0, ge=0, le=200, description="0 = choose automatically")

    # Trends & incidents
    trend_freq: Literal["h", "D", "W"] = "D"
    spike_z: float = 3.0
    spike_min_count: int = 5
    spike_window: int = 7
    duplicate_threshold: float = Field(0.85, gt=0.0, le=1.0)
    incident_min_size: int = 8
    incident_window_hours: int = 48
    incident_burst_ratio: float = 4.0
    incident_merge_threshold: float = 0.6

    # Ingestion / storage
    redact_pii: bool = True
    max_rows: int = 50_000
    max_upload_mb: int = 25
    session_dir: Path = Path(".sessions")
    session_cache_size: int = 8

    # UI
    ui_mode: Literal["local", "api"] = "local"
    api_url: str = "http://localhost:8000"


@lru_cache
def get_settings() -> Settings:
    return Settings()
