"""Loading of the NASA C-MAPSS turbofan degradation dataset.

The raw data consists of four sub-datasets (FD001-FD004), each shipped as three
whitespace-separated text files::

    train_FD00X.txt   run-to-failure trajectories (one row per engine cycle)
    test_FD00X.txt    trajectories truncated at an unknown time before failure
    RUL_FD00X.txt     true RUL at the last cycle of every test engine

Usage (download helper)::

    python -m src.data.load_data --download
"""

from __future__ import annotations

import argparse
import io
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Union

import pandas as pd

from src.utils import get_logger, load_config, resolve_path

logger = get_logger(__name__)

OPERATING_SETTINGS = ["op_setting_1", "op_setting_2", "op_setting_3"]
SENSORS = [f"s{i}" for i in range(1, 22)]
COLUMNS = ["unit", "cycle"] + OPERATING_SETTINGS + SENSORS

#: Physical meaning of every sensor (Saxena et al., 2008).
SENSOR_DESCRIPTIONS: Dict[str, str] = {
    "s1": "T2 - Total temperature at fan inlet (°R)",
    "s2": "T24 - Total temperature at LPC outlet (°R)",
    "s3": "T30 - Total temperature at HPC outlet (°R)",
    "s4": "T50 - Total temperature at LPT outlet (°R)",
    "s5": "P2 - Pressure at fan inlet (psia)",
    "s6": "P15 - Total pressure in bypass-duct (psia)",
    "s7": "P30 - Total pressure at HPC outlet (psia)",
    "s8": "Nf - Physical fan speed (rpm)",
    "s9": "Nc - Physical core speed (rpm)",
    "s10": "epr - Engine pressure ratio (P50/P2)",
    "s11": "Ps30 - Static pressure at HPC outlet (psia)",
    "s12": "phi - Ratio of fuel flow to Ps30 (pps/psi)",
    "s13": "NRf - Corrected fan speed (rpm)",
    "s14": "NRc - Corrected core speed (rpm)",
    "s15": "BPR - Bypass ratio",
    "s16": "farB - Burner fuel-air ratio",
    "s17": "htBleed - Bleed enthalpy",
    "s18": "Nf_dmd - Demanded fan speed (rpm)",
    "s19": "PCNfR_dmd - Demanded corrected fan speed (rpm)",
    "s20": "W31 - HPT coolant bleed (lbm/s)",
    "s21": "W32 - LPT coolant bleed (lbm/s)",
}

#: Scenario description of each sub-dataset.
DATASET_INFO: Dict[str, Dict[str, Union[int, str]]] = {
    "FD001": {"conditions": 1, "fault_modes": 1, "faults": "HPC degradation"},
    "FD002": {"conditions": 6, "fault_modes": 1, "faults": "HPC degradation"},
    "FD003": {"conditions": 1, "fault_modes": 2, "faults": "HPC + Fan degradation"},
    "FD004": {"conditions": 6, "fault_modes": 2, "faults": "HPC + Fan degradation"},
}

#: Known public locations of the original NASA archive (tried in order).
DOWNLOAD_URLS = [
    "https://phm-datasets.s3.amazonaws.com/NASA/"
    "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip",
    "https://data.nasa.gov/download/ff5v-kuh6/application%2Fzip",
]


@dataclass
class CMAPSSDataset:
    """Container for one C-MAPSS sub-dataset."""

    name: str
    train: pd.DataFrame
    test: pd.DataFrame
    test_rul: pd.Series  # indexed by unit

    @property
    def n_train_engines(self) -> int:
        return int(self.train["unit"].nunique())

    @property
    def n_test_engines(self) -> int:
        return int(self.test["unit"].nunique())


def read_cmapss_file(path: Union[str, Path, io.TextIOBase]) -> pd.DataFrame:
    """Read one raw C-MAPSS trajectory file into a tidy DataFrame.

    Rows are sorted by ``(unit, cycle)`` and the index is reset, which every
    downstream grouped transformation relies on.
    """
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS, engine="python")
    df = df.dropna(axis=1, how="all")
    df[["unit", "cycle"]] = df[["unit", "cycle"]].astype(int)
    return df.sort_values(["unit", "cycle"]).reset_index(drop=True)


def read_sensor_table(path_or_buffer: Union[str, Path, io.IOBase]) -> pd.DataFrame:
    """Read user-supplied sensor data, either raw C-MAPSS text or a CSV.

    A CSV must contain the canonical column names (``unit``, ``cycle``,
    ``op_setting_1..3``, ``s1..s21``); a headerless file is parsed as raw text.
    """
    raw = path_or_buffer.read() if hasattr(path_or_buffer, "read") else Path(
        path_or_buffer).read_bytes()
    text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    first_line = text.lstrip().splitlines()[0] if text.strip() else ""
    if "unit" in first_line and "cycle" in first_line:
        df = pd.read_csv(io.StringIO(text))
        missing = set(COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in uploaded file: {sorted(missing)}")
        df = df[COLUMNS].copy()
        df[["unit", "cycle"]] = df[["unit", "cycle"]].astype(int)
        return df.sort_values(["unit", "cycle"]).reset_index(drop=True)
    return read_cmapss_file(io.StringIO(text))


def load_dataset(name: str, raw_dir: Optional[Union[str, Path]] = None) -> CMAPSSDataset:
    """Load the train / test / RUL files of one sub-dataset (e.g. ``"FD001"``)."""
    if name not in DATASET_INFO:
        raise ValueError(f"Unknown dataset {name!r}; expected one of {list(DATASET_INFO)}")
    raw = Path(raw_dir) if raw_dir else resolve_path(load_config()["paths"]["raw_dir"])
    files = {k: raw / f"{k}_{name}.txt" for k in ("train", "test", "RUL")}
    absent = [str(p) for p in files.values() if not p.exists()]
    if absent:
        raise FileNotFoundError(
            "C-MAPSS files not found: " + ", ".join(absent)
            + "\nRun `python -m src.data.load_data --download` or see README.md."
        )
    train = read_cmapss_file(files["train"])
    test = read_cmapss_file(files["test"])
    rul = pd.read_csv(files["RUL"], sep=r"\s+", header=None, engine="python").iloc[:, 0]
    rul.index = pd.RangeIndex(1, len(rul) + 1, name="unit")
    rul.name = "rul"
    logger.info("Loaded %s: %d train engines (%d rows), %d test engines (%d rows)",
                name, train["unit"].nunique(), len(train),
                test["unit"].nunique(), len(test))
    return CMAPSSDataset(name=name, train=train, test=test, test_rul=rul.astype(int))


def load_all(names: Iterable[str], raw_dir: Optional[Path] = None) -> Dict[str, CMAPSSDataset]:
    """Load several sub-datasets at once."""
    return {n: load_dataset(n, raw_dir) for n in names}


def _extract_cmapss(archive: zipfile.ZipFile, target: Path) -> int:
    """Extract every C-MAPSS text file, descending into nested zip archives."""
    count = 0
    for member in archive.namelist():
        base = Path(member).name
        if base.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(archive.read(member))) as inner:
                count += _extract_cmapss(inner, target)
        elif base.endswith(".txt") and base.split("_")[0] in {"train", "test", "RUL"}:
            (target / base).write_bytes(archive.read(member))
            count += 1
    return count


def download_cmapss(raw_dir: Optional[Path] = None, timeout: int = 120) -> Path:
    """Download and extract the official NASA archive into ``data/raw``."""
    target = Path(raw_dir) if raw_dir else resolve_path(load_config()["paths"]["raw_dir"])
    target.mkdir(parents=True, exist_ok=True)
    for url in DOWNLOAD_URLS:
        try:
            logger.info("Downloading C-MAPSS from %s", url)
            with urllib.request.urlopen(url, timeout=timeout) as response:
                payload = response.read()
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                n_files = _extract_cmapss(archive, target)
            if n_files >= 12:
                logger.info("Extracted %d files into %s", n_files, target)
                return target
            logger.warning("Archive from %s contained only %d files", url, n_files)
        except Exception as exc:  # noqa: BLE001 - try the next mirror
            logger.warning("Download failed (%s): %s", url, exc)
    raise RuntimeError(
        "Automatic download failed. Download the 'Turbofan Engine Degradation "
        "Simulation' dataset manually from the NASA PCoE data repository and "
        f"copy the 12 .txt files into {target}"
    )


def _main() -> None:
    parser = argparse.ArgumentParser(description="C-MAPSS data utilities")
    parser.add_argument("--download", action="store_true", help="download the raw data")
    parser.add_argument("--check", action="store_true", help="load and summarise every subset")
    args = parser.parse_args()
    if args.download:
        download_cmapss()
    if args.check or not args.download:
        for name in DATASET_INFO:
            ds = load_dataset(name)
            print(f"{name}: train={ds.n_train_engines} engines, test={ds.n_test_engines} engines")


if __name__ == "__main__":
    _main()
