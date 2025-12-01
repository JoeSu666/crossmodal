import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from PIL import Image
import random
from sklearn.model_selection import train_test_split, StratifiedKFold
import pytorch_lightning as pl

class BaseHistoDataset(Dataset):
    """Base dataset class for histopathology image datasets."""
    
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # These should be set by child classes
        self.class_names = []
        self.num_classes = 0
        self.class_to_idx = {}
        self.samples = []
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        try:
            image = torch.load(img_path)
        except Exception as e:
            raise RuntimeError(f"Error loading image {img_path}: {e}")
        
        # image = image.float()
        # image = image.permute(0, 3, 1, 2)  # Convert to CxHxW format
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_distribution(self):
        """Get the distribution of classes in the current split"""
        class_counts = {}
        for _, label in self.samples:
            class_name = self.class_names[label]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        return class_counts

class HER2(BaseHistoDataset):
    """HER2 Dataset for histopathology image classification."""

    def __init__(self, root_dir, split='train', splitdir=None, fold=0, transform=None):
        super().__init__(root_dir, split, transform)
        self.splitdir = splitdir
        self.fold = fold

        # Define class names (2 classes)
        self.class_names = ['negative', 'positive']
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)
        
        # Load dataset
        self.samples = self._load_dataset()
        
    def _load_dataset(self):
        """Load dataset based on split"""
        splitname = os.path.join(self.splitdir, f'splits_{self.fold}.csv')
        splitdf = pd.read_csv(splitname, dtype={f'{self.split}': str})
        samplelist = splitdf[self.split].dropna().tolist()
        labellist = splitdf[f'{self.split}_label'].dropna().tolist()

        samplelist = [os.path.join(self.root_dir, sample+'.pt') for sample in samplelist]
        labellist = [0 if label == 'negative' else 1 for label in labellist]

        return list(zip(samplelist, labellist))

    def get_weights(self):
        # get weights for weight random sampler (training only)
        splitname = os.path.join(self.splitdir, f'splits_{self.fold}.csv')
        splitdf = pd.read_csv(splitname, dtype={f'{self.split}': str})
        traindf = splitdf[['train', 'train_label']].dropna().reset_index(drop=True)
        traindf.set_index('train')

        N = len(traindf)
        w_per_cls = {'positive': N/(traindf['train_label']=='positive').sum(), 'negative': N/(traindf['train_label']=='negative').sum()}

        weights = [w_per_cls[traindf.loc[name, 'train_label']] for name in traindf.index.tolist()]

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

