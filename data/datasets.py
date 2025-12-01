import os
from typing import List, Sequence, Tuple

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

class BaseHistoDataset(Dataset):
    """Base dataset class for histopathology image datasets."""

    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        # These should be set by child classes
        self.class_names: List[str] = []
        self.num_classes = 0
        self.class_to_idx = {}
        self.samples: List[Tuple[Sequence[str], int]] = []

    def __len__(self):
        return len(self.samples)

    def _load_tensor(self, tensor_path: str) -> torch.Tensor:
        try:
            return torch.load(tensor_path)
        except Exception as exc:
            raise RuntimeError(f"Error loading tensor {tensor_path}: {exc}") from exc

    def __getitem__(self, idx):
        file_paths, label = self.samples[idx]

        if isinstance(file_paths, (list, tuple)) and len(file_paths) == 2:
            he_path, ihc_path = file_paths
            he_tensor = self._load_tensor(he_path)
            ihc_tensor = self._load_tensor(ihc_path)

            if self.transform:
                he_tensor = self.transform(he_tensor)
                ihc_tensor = self.transform(ihc_tensor)

            return (he_tensor, ihc_tensor), label

        he_tensor = self._load_tensor(file_paths)
        if self.transform:
            he_tensor = self.transform(he_tensor)

        return he_tensor, label

    def get_class_distribution(self):
        """Get the distribution of classes in the current split"""
        class_counts = {}
        for _, label in self.samples:
            class_name = self.class_names[label]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        return class_counts

class HER2(BaseHistoDataset):
    """HER2 Dataset for histopathology image classification."""

    def __init__(self, root_dir, split="train", splitdir=None, fold=0, transform=None):
        super().__init__(root_dir, split, transform)
        self.splitdir = splitdir
        self.fold = fold

        # Define class names (2 classes)
        self.class_names = [0, 1, 2, 3]
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)

        # Load dataset
        self.samples = self._load_dataset()

    def _split_sample_names(self, raw_entry: str) -> Tuple[str, ...]:
        """Split a raw sample string into individual filenames.

        Accepts comma- or whitespace-separated names to make the split file
        format flexible.
        """

        if "," in raw_entry:
            names = [name.strip() for name in raw_entry.split(",") if name.strip()]
        else:
            names = [name.strip() for name in raw_entry.split() if name.strip()]

        return tuple(names)


    def _load_dataset(self):
        """Load dataset based on split."""

        splitname = os.path.join(self.splitdir, f"splits_{self.fold}.csv")
        splitdf = pd.read_csv(splitname, dtype={f"{self.split}": str})

        samplelist = splitdf[self.split].dropna().tolist()
        labellist = splitdf[f"{self.split}_label"].dropna().tolist()

        parsed_samples: List[Tuple[Sequence[str], int]] = []

        for raw_sample, label in zip(samplelist, labellist):
            names = self._split_sample_names(raw_sample)

            if self.split == "train":
                if len(names) != 2:
                    raise ValueError(
                        "Training split entries must contain two filenames (HE and IHC). "
                        "Use a comma or whitespace to separate them."
                    )
                he_name, ihc_name = names
                he_path = os.path.join(self.root_dir, f"{he_name}.pt")
                ihc_path = os.path.join(self.root_dir, f"{ihc_name}.pt")
                parsed_samples.append(((he_path, ihc_path), label))
            else:
                if len(names) != 1:
                    raise ValueError(
                        f"{self.split} split entries must contain exactly one HE filename."
                    )
                he_path = os.path.join(self.root_dir, f"{names[0]}.pt")
                parsed_samples.append((he_path, label))

        return parsed_samples

    def get_weights(self):
        # get weights for weight random sampler (training only)
        if self.split != "train":
            raise RuntimeError("Class weights are only defined for the training split.")

        label_counts = {}
        for _, label in self.samples:
            label_counts[label] = label_counts.get(label, 0) + 1

        total = len(self.samples)
        weights = []
        for _, label in self.samples:
            class_weight = total / label_counts[label]
            weights.append(class_weight)

        return torch.DoubleTensor(weights)



class DataModule(pl.LightningDataModule):
    """
    Generic PyTorch Lightning DataModule for histopathology datasets
    """
    
    def __init__(self, data_dir, dataset_class, batch_size=32,
                 split_dir=None, fold=0, num_workers=4, weighted_sampler=False):
        super().__init__()
        self.data_dir = data_dir
        self.dataset_class = dataset_class
        self.batch_size = batch_size
        self.split_dir = split_dir
        self.fold = fold
        self.num_workers = num_workers
        self.weighted_sampler = weighted_sampler
        
        # Initialize transforms
        # self.transform = transforms.Compose([
        #     # transforms.Resize((self.image_size, self.image_size)),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ])
    
    def setup(self, stage=None):
        """Setup datasets for each stage"""
        if stage == 'fit' or stage is None:
            self.train_dataset = self.dataset_class(
                root_dir=self.data_dir,
                split='train',
                splitdir=self.split_dir,
                fold=self.fold,
                transform=None
            )
            
            self.val_dataset = self.dataset_class(
                root_dir=self.data_dir,
                split='val',
                splitdir=self.split_dir,
                fold=self.fold,
                transform=None
            )
        
        if stage == 'test' or stage is None:
            self.test_dataset = self.dataset_class(
                root_dir=self.data_dir,
                split='test',
                splitdir=self.split_dir,
                fold=self.fold,
                transform=None
            )
    
    def train_dataloader(self):
        if self.weighted_sampler:
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=True if self.num_workers > 0 else False,
                sampler=torch.utils.data.WeightedRandomSampler(
                    weights=self.train_dataset.get_weights(),
                    num_samples=len(self.train_dataset)
                )
            )
        else:
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=True if self.num_workers > 0 else False,
            )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def predict_dataloader(self):
        return self.test_dataloader()

