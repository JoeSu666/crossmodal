from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class CrossModalDataset(Dataset):
    """Dataset for paired HE/IHC embeddings during training and HE-only inference.

    Args:
        split_file: CSV file containing a column with paths and a ``label`` column.
        split_column: Name of the column that stores one or two embedding paths.
        root_dir: Optional directory that is prepended to each path entry.

    Notes:
        * Training rows should include both HE and IHC filenames in ``split_column``.
        * Validation/test rows can provide only the HE filename.
        * Path tokens are split on commas or whitespace; using commas is recommended
          to avoid ambiguity.
    """

    def __init__(
        self,
        split_file: Union[str, Path],
        split_column: str = "train",
        root_dir: Union[str, Path, None] = None,
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir) if root_dir is not None else None

        df = pd.read_csv(split_file)
        if split_column not in df.columns:
            raise ValueError(f"Missing column '{split_column}' in {split_file}")
        if "label" not in df.columns:
            raise ValueError("Dataset file must contain a 'label' column")

        self.samples: List[Tuple[Path, Optional[Path], int]] = []
        for _, row in df.iterrows():
            raw_paths = str(row[split_column])
            he_path, ihc_path = self._parse_paths(raw_paths)
            label = int(row["label"])
            self.samples.append((he_path, ihc_path, label))

    def _parse_paths(self, raw_paths: str) -> Tuple[Path, Optional[Path]]:
        tokens = [p for p in re.split(r"[,\s]+", raw_paths.strip()) if p]
        if not tokens:
            raise ValueError("Path column contains an empty entry")

        he_path: Optional[Path] = None
        ihc_path: Optional[Path] = None
        for token in tokens:
            path = Path(token)
            if self.root_dir is not None and not path.is_absolute():
                path = self.root_dir / path

            lower_token = token.lower()
            if "ihc" in lower_token and ihc_path is None:
                ihc_path = path
            elif "he" in lower_token and he_path is None:
                he_path = path
            elif he_path is None:
                he_path = path
            elif ihc_path is None:
                ihc_path = path

        if he_path is None:
            raise ValueError(f"Unable to locate HE path from entry: '{raw_paths}'")

        return he_path, ihc_path

    @staticmethod
    def _load_embedding(path: Path) -> torch.Tensor:
        suffix = path.suffix.lower()
        if suffix in {".pt", ".pth"}:
            emb = torch.load(path)
        elif suffix == ".npy":
            emb = np.load(path)
        else:
            emb = torch.load(path)

        if isinstance(emb, np.ndarray):
            emb = torch.from_numpy(emb)
        return emb.float()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        he_path, ihc_path, label = self.samples[index]
        he_emb = self._load_embedding(he_path)

        if ihc_path is not None:
            ihc_emb = self._load_embedding(ihc_path)
            return (he_emb, ihc_emb), label

        return he_emb, label
