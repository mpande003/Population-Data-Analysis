"""Dataset download and loading for the IndiaSet Census 2011 dataset.

Only https://huggingface.co/datasets/indiaset/census-2011 is used as a data source.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

REPO_ID = "indiaset/census-2011"
LOCAL_DATA_DIR = Path(__file__).parent / "data"
LOCAL_EXTS = (".parquet", ".csv", ".xlsx", ".xls")


def list_repo_files(repo_id: str = REPO_ID) -> list[str]:
    """Inspect the Hugging Face repository file list (no filename is assumed)."""
    from huggingface_hub import HfApi
    return sorted(HfApi().list_repo_files(repo_id, repo_type="dataset"))


def choose_data_file(files: list[str]) -> str:
    """Pick a data file from the verified repo listing (Parquet preferred)."""
    parquet = [f for f in files if f.lower().endswith(".parquet")]
    if parquet:
        train = [f for f in parquet if "train" in f.lower()]
        return (train or parquet)[0]
    for ext in (".csv", ".xlsx"):
        hits = [f for f in files if f.lower().endswith(ext)]
        if hits:
            return hits[0]
    raise FileNotFoundError(f"No Parquet/CSV/Excel file in repo. Files found: {files}")


def read_any(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    s = path.suffix.lower()
    if s == ".parquet":
        return pd.read_parquet(path, engine="pyarrow")
    if s == ".csv":
        return pd.read_csv(path)
    if s in (".xlsx", ".xls"):
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {path}")


def download_dataset(repo_id: str = REPO_ID) -> tuple[pd.DataFrame, str]:
    from huggingface_hub import hf_hub_download
    files = list_repo_files(repo_id)
    fname = choose_data_file(files)
    local = hf_hub_download(repo_id=repo_id, filename=fname, repo_type="dataset")
    return read_any(local), f"Hugging Face: {repo_id}/{fname}"


def load_local(local_dir: str | Path = LOCAL_DATA_DIR) -> tuple[pd.DataFrame, str]:
    local_dir = Path(local_dir)
    cands = sorted(p for p in local_dir.glob("*") if p.suffix.lower() in LOCAL_EXTS)
    if not cands:
        raise FileNotFoundError(
            f"No local data file found in {local_dir}. Download the Parquet file from "
            f"https://huggingface.co/datasets/{REPO_ID} and place it there.")
    return read_any(cands[0]), f"Local file: {cands[0].name}"


def load_raw(prefer_local: bool = True) -> tuple[pd.DataFrame, str]:
    """Return (raw_dataframe, source_description). Tries download, then local fallback."""
    errors = []
    order = (load_local, download_dataset) if prefer_local else (download_dataset, load_local)
    for fn in order:
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - report every failure transparently
            errors.append(f"{fn.__name__}: {exc}")
    raise RuntimeError("Dataset could not be loaded.\n" + "\n".join(errors))
