"""Dataset loaders for the three benchmarks in Table 1.

Each benchmark can be loaded from disk (if the raw CSV/parquet is present)
or generated synthetically for unit testing and rapid prototyping.

+------------------+--------+----+-------+----------------------------+
| Dataset          |      T |  D | Freq  | Target                     |
+------------------+--------+----+-------+----------------------------+
| S&P 500          |   5034 | 25 | daily | 21-day realised volatility |
| UCI Appliances   |  19735 | 28 | 10min | Energy (Wh)                |
| Jena Climate     | 420551 | 14 | 10min | Temperature (+6 h)         |
+------------------+--------+----+-------+----------------------------+
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset specifications (Table 1)
# ---------------------------------------------------------------------------

_SPECS: dict[str, dict] = {
    "sp500": {
        "T": 5034,
        "D": 25,
        "freq": "daily",
        "target_name": "realized_vol_21d",
        "description": "S&P 500 daily -- target is 21-day realised volatility",
    },
    "uci_appliances": {
        "T": 19735,
        "D": 28,
        "freq": "10min",
        "target_name": "energy_wh",
        "description": "UCI Appliances Energy -- target is Energy (Wh)",
    },
    "jena_climate": {
        "T": 420551,
        "D": 14,
        "freq": "10min",
        "target_name": "temperature_6h",
        "description": "Jena Climate -- target is Temperature (+6 h ahead)",
    },
}

DatasetName = Literal["sp500", "uci_appliances", "jena_climate"]


@dataclass
class DatasetLoader:
    """Load or generate benchmark datasets for RUC-TS evaluation.

    Parameters
    ----------
    data_dir : str or Path, optional
        Root directory where raw data files are stored.  Each dataset is
        expected under ``<data_dir>/<dataset_name>/``.
    seed : int
        Random seed used by ``generate_synthetic``.
    """

    data_dir: Path = Path("data")
    seed: int = 42

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def available_datasets() -> list[str]:
        """Return the names of all supported benchmark datasets."""
        return list(_SPECS.keys())

    @staticmethod
    def dataset_info(name: DatasetName) -> dict:
        """Return the specification dict for *name*."""
        if name not in _SPECS:
            raise ValueError(
                f"Unknown dataset '{name}'. Choose from {list(_SPECS.keys())}."
            )
        return dict(_SPECS[name])

    def load(self, name: DatasetName) -> tuple[pd.DataFrame, pd.Series]:
        """Load a real dataset from ``data_dir``.

        Parameters
        ----------
        name : str
            One of ``"sp500"``, ``"uci_appliances"``, ``"jena_climate"``.

        Returns
        -------
        features : pd.DataFrame
            Feature matrix of shape ``(T, D)``.
        target : pd.Series
            Target vector of length ``T``.
        """
        spec = self.dataset_info(name)

        if name == "sp500":
            return self._load_sp500(spec)
        elif name == "uci_appliances":
            return self._load_uci_appliances(spec)
        elif name == "jena_climate":
            return self._load_jena_climate(spec)
        else:
            raise ValueError(f"Unknown dataset '{name}'.")

    def generate_synthetic(
        self,
        name: DatasetName,
        *,
        scale: float = 1.0,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Create realistic synthetic data matching a benchmark's dimensions.

        The synthetic data preserves the dimensionality (T, D), basic
        autocorrelation structure, and target-feature correlations so that
        downstream code can be tested without the real data files.

        Parameters
        ----------
        name : str
            Which benchmark to mimic.
        scale : float
            Fraction of original T to generate (e.g. 0.1 for a 10 % subset).
            Useful for quick smoke tests.

        Returns
        -------
        features : pd.DataFrame
        target : pd.Series
        """
        spec = self.dataset_info(name)
        T = max(10, int(spec["T"] * scale))
        D = spec["D"]
        rng = np.random.default_rng(self.seed)

        if name == "sp500":
            features, target = self._synth_sp500(T, D, rng)
        elif name == "uci_appliances":
            features, target = self._synth_uci_appliances(T, D, rng)
        elif name == "jena_climate":
            features, target = self._synth_jena_climate(T, D, rng)
        else:
            raise ValueError(f"Unknown dataset '{name}'.")

        logger.info(
            "Generated synthetic '%s': T=%d, D=%d.", name, len(target), D
        )
        return features, target

    # ------------------------------------------------------------------
    # Real-data loaders (stub implementations -- fill paths as needed)
    # ------------------------------------------------------------------

    def _load_sp500(self, spec: dict) -> tuple[pd.DataFrame, pd.Series]:
        """Load S&P 500 data from CSV / parquet."""
        path = self.data_dir / "sp500" / "sp500.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"S&P 500 data not found at {path}. "
                "Use generate_synthetic('sp500') for testing."
            )
        df = pd.read_csv(path, parse_dates=["date"], index_col="date")
        target = df[spec["target_name"]]
        features = df.drop(columns=[spec["target_name"]])
        return features, target

    def _load_uci_appliances(self, spec: dict) -> tuple[pd.DataFrame, pd.Series]:
        """Load UCI Appliances Energy Prediction data."""
        path = self.data_dir / "uci_appliances" / "energydata_complete.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"UCI Appliances data not found at {path}. "
                "Use generate_synthetic('uci_appliances') for testing."
            )
        df = pd.read_csv(path, parse_dates=["date"], index_col="date")
        target = df["Appliances"].rename(spec["target_name"])
        features = df.drop(columns=["Appliances"])
        return features, target

    def _load_jena_climate(self, spec: dict) -> tuple[pd.DataFrame, pd.Series]:
        """Load Jena Climate dataset."""
        path = self.data_dir / "jena_climate" / "jena_climate_2009_2016.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Jena Climate data not found at {path}. "
                "Use generate_synthetic('jena_climate') for testing."
            )
        df = pd.read_csv(path, parse_dates=["Date Time"], index_col="Date Time")
        # Target: temperature 6 hours (36 steps of 10 min) ahead
        df[spec["target_name"]] = df["T (degC)"].shift(-36)
        df = df.dropna(subset=[spec["target_name"]])
        target = df[spec["target_name"]]
        features = df.drop(columns=[spec["target_name"]])
        return features, target

    # ------------------------------------------------------------------
    # Synthetic generators
    # ------------------------------------------------------------------

    def _synth_sp500(
        self, T: int, D: int, rng: np.random.Generator
    ) -> tuple[pd.DataFrame, pd.Series]:
        """S&P 500 synthetic: correlated financial returns + realised vol."""
        dates = pd.bdate_range(end="2024-12-31", periods=T, freq="B")

        # Simulate log-returns with a factor structure
        n_factors = min(5, D)
        factors = rng.standard_normal((T, n_factors)) * 0.01
        loadings = rng.standard_normal((n_factors, D)) * 0.5
        idiosyncratic = rng.standard_normal((T, D)) * 0.005
        returns = factors @ loadings + idiosyncratic

        # Derived features: rolling means, rolling stds, momentum
        feature_names = [f"return_{i}" for i in range(D)]
        features = pd.DataFrame(returns, index=dates, columns=feature_names)

        # Target: 21-day realised volatility of a synthetic index
        index_return = returns.mean(axis=1)
        realized_vol = (
            pd.Series(index_return, index=dates)
            .rolling(21)
            .std()
            .bfill()
        )
        realized_vol.name = "realized_vol_21d"

        return features, realized_vol

    def _synth_uci_appliances(
        self, T: int, D: int, rng: np.random.Generator
    ) -> tuple[pd.DataFrame, pd.Series]:
        """UCI Appliances synthetic: periodic signals + energy target."""
        timestamps = pd.date_range(
            start="2016-01-11", periods=T, freq="10min"
        )
        t = np.arange(T, dtype=np.float64)

        columns: dict[str, np.ndarray] = {}
        for i in range(D):
            # Mix of daily and weekly periodicities with noise
            period = 144 * (1 + i % 7)  # 144 = 24h in 10-min steps
            phase = rng.uniform(0, 2 * np.pi)
            amplitude = rng.uniform(0.5, 5.0)
            noise = rng.normal(0, 0.3, T)
            columns[f"sensor_{i}"] = amplitude * np.sin(2 * np.pi * t / period + phase) + noise

        features = pd.DataFrame(columns, index=timestamps)

        # Energy target: non-negative, depends on a few sensors + time-of-day
        base = 50 + 10 * np.sin(2 * np.pi * t / 144)  # daily cycle
        linear_combo = sum(
            rng.uniform(-0.5, 1.5) * columns[f"sensor_{i}"]
            for i in range(min(5, D))
        )
        energy = np.abs(base + linear_combo + rng.normal(0, 5, T))
        target = pd.Series(energy, index=timestamps, name="energy_wh")

        return features, target

    def _synth_jena_climate(
        self, T: int, D: int, rng: np.random.Generator
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Jena Climate synthetic: weather-like signals + 6h temp forecast."""
        timestamps = pd.date_range(
            start="2009-01-01", periods=T, freq="10min"
        )
        t = np.arange(T, dtype=np.float64)

        # Daily period = 144 steps, yearly ~ 52560 steps
        daily = np.sin(2 * np.pi * t / 144)
        yearly = np.sin(2 * np.pi * t / 52560)

        weather_vars = [
            "temperature",
            "pressure",
            "humidity",
            "wind_speed",
            "wind_dir",
            "dew_point",
            "visibility",
            "cloud_cover",
            "precip",
            "solar_rad",
            "uv_index",
            "ozone",
            "pm25",
            "co2",
        ][:D]

        columns: dict[str, np.ndarray] = {}
        for i, name in enumerate(weather_vars):
            amp_d = rng.uniform(2, 10)
            amp_y = rng.uniform(5, 15)
            offset = rng.uniform(-5, 25)
            ar_noise = np.empty(T)
            ar_noise[0] = rng.normal(0, 1)
            for step in range(1, T):
                ar_noise[step] = 0.95 * ar_noise[step - 1] + rng.normal(0, 0.5)
            columns[name] = offset + amp_d * daily + amp_y * yearly + ar_noise

        # Pad if D > len(weather_vars)
        for i in range(len(weather_vars), D):
            columns[f"extra_{i}"] = rng.normal(0, 1, T)

        features = pd.DataFrame(columns, index=timestamps)

        # Target: temperature 6 hours (36 steps) ahead
        temp = columns[weather_vars[0]]
        target_values = np.empty(T)
        target_values[:T - 36] = temp[36:]
        target_values[T - 36:] = np.nan
        target = pd.Series(target_values, index=timestamps, name="temperature_6h")

        # Drop NaN rows at the end
        valid = target.dropna().index
        features = features.loc[valid]
        target = target.loc[valid]

        return features, target
