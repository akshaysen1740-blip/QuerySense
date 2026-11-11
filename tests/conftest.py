from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from signalsift.config import Settings
from signalsift.datasets import generate_tickets
from signalsift.pipeline import Analysis, analyze


@pytest.fixture(scope="session")
def tickets() -> pd.DataFrame:
    return generate_tickets(n=2000, seed=3)


@pytest.fixture(scope="session")
def settings(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    return Settings(session_dir=Path(tmp_path_factory.mktemp("sessions")))


@pytest.fixture(scope="session")
def analysis(tickets: pd.DataFrame, settings: Settings) -> Analysis:
    return analyze(tickets, settings, "abcdef123456", name="test")
