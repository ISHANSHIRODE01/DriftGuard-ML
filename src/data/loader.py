"""
ELEC2 loading and strictly temporal batching.

ELEC2 (Harries, 1999) is the standard concept-drift benchmark: 45,312
half-hourly records from the New South Wales electricity market. The label is
whether the spot price moved UP or DOWN relative to a 24-hour moving average.

The reason it is a *benchmark* and not just a dataset: market rules changed
partway through the collection period, so the relationship between features
and label genuinely shifts. A monitoring system evaluated on it is being
tested against real concept drift rather than injected noise.

Batching is strictly chronological. Shuffling would leak future information
backwards and make the whole drift experiment meaningless.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

FEATURES = ["nswprice", "nswdemand", "vicprice", "vicdemand", "transfer", "period", "day"]
TARGET = "target"

# Features monitored for drift. `day` and `period` are cyclical calendar
# fields -- they shift by construction across batches and would produce
# permanent false-positive drift, so they are excluded from monitoring while
# remaining available to the model.
MONITORED_FEATURES = ["nswprice", "nswdemand", "vicprice", "vicdemand", "transfer"]


@dataclass
class Batch:
    index: int
    X: pd.DataFrame
    y: pd.Series
    raw: pd.DataFrame

    def __len__(self) -> int:
        return len(self.raw)


def load_elec2(path: str | Path = "data/elec2.csv") -> pd.DataFrame:
    """Load ELEC2 preserving row order (chronological)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Run `python scripts/download_data.py` first."
        )
    df = pd.read_csv(p)

    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"ELEC2 is missing expected columns: {sorted(missing)}")

    # Deterministic dtypes -- the drift detector routes on dtype.
    for c in FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[TARGET] = df[TARGET].astype(int)

    return df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)


def make_batches(
    df: pd.DataFrame, batch_size: int = 1440, max_batches: Optional[int] = None
) -> list[Batch]:
    """
    Split chronologically into fixed-size batches.

    batch_size=1440 is 30 days of half-hourly readings (48 * 30), so each
    batch is roughly one month of production traffic.
    """
    batches: list[Batch] = []
    n = len(df) // batch_size
    if max_batches is not None:
        n = min(n, max_batches)

    for i in range(n):
        chunk = df.iloc[i * batch_size : (i + 1) * batch_size]
        batches.append(
            Batch(
                index=i,
                X=chunk[FEATURES].reset_index(drop=True),
                y=chunk[TARGET].reset_index(drop=True),
                raw=chunk.reset_index(drop=True),
            )
        )
    return batches


def stream_batches(
    df: pd.DataFrame, batch_size: int = 1440
) -> Iterator[Batch]:
    """Generator form, for simulating a live feed."""
    for b in make_batches(df, batch_size):
        yield b
