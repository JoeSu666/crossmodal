#!/usr/bin/env python3
"""
End-to-End MIL training script for WSIs with Cross-Validation support
"""

import argparse
import os
import shutil
import yaml
from datetime import datetime

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

import data.datasets as datasets
from data.datasets import DataModule
from models.model_interface import ModelInterface

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_attribute(module, name):
    """Simple getattr wrapper with a clear error message."""
    try:
        return getattr(module, name)
    except AttributeError as exc:
        raise ValueError(f"{name} is not available in the provided module.") from exc


def create_experiment_name(config, model_name, ckpt_path=None, fold=None, tag=None):
    """Create a unique experiment name"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_name = "scratch" if ckpt_path is None else os.path.basename(ckpt_path).replace('.pth', '')
    fold_suffix = f"_fold{fold}" if fold is not None else ""
    tag_suffix = f"_{tag}" if tag else ""
    return f"{model_name}_{ckpt_name}{fold_suffix}{tag_suffix}_{timestamp}"


def setup_callbacks(config):
    """Setup training callbacks"""
    callbacks = []

    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        monitor=config['training']['monitor'],
        mode=config['training']['mode'],
        save_top_k=config['training']['save_top_k'],
        save_last=config['training']['save_last'],
        filename='best',
        auto_insert_metric_name=False
    )
    callbacks.append(checkpoint_callback)

    # Early stopping
    early_stopping = EarlyStopping(
        monitor=config['training']['monitor'],
        mode=config['training']['mode'],
        patience=config['training']['patience'],
        verbose=True
    )
    callbacks.append(early_stopping)

    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)

    return callbacks


def setup_logger(config, experiment_name, output_dir):
    """Setup experiment logger"""
    if config['logging']['use_wandb']:
        logger = WandbLogger(
            project=config['logging']['project_name'],
            name=experiment_name,
            save_dir=os.path.join(output_dir, 'logs')
        )
    else:
        logger = TensorBoardLogger(
            save_dir=config['logging']['log_dir'],
            name=config['logging']['project_name'],
            version=experiment_name
        )

    return logger


def train_single_fold(config, model_name, fold, tag, output_dir='./outputs'):
    """Train a single fold of the cross-validation"""

    # Create experiment name for this fold
    experiment_name = create_experiment_name(config, model_name, fold=fold, tag=tag)

    print(f"\n{'='*60}")
    print(f"Training Fold {fold}: {experiment_name}")
    print(f"Model: {model_name}")
    print("Checkpoint: None (training from scratch)")
    print(f"{'='*60}")

    # Setup data module with specific fold
    dataset_class = get_attribute(datasets, config['data']['dataset_class'])
    data_module = DataModule(
        data_dir=config['data']['data_dir'],
        dataset_class=dataset_class,
        batch_size=config['data']['batch_size'],
        split_dir=config['data']['split_dir'],
        fold=fold,  # Use the specific fold
        num_workers=config['data']['num_workers'],
        weighted_sampler=True
    )

    # Setup model
    model = ModelInterface(
        num_classes=config['model']['num_classes'],
        learning_rate=config['model']['learning_rate'],
        weight_decay=config['model']['weight_decay'],
        scheduler=config['model']['scheduler'],
        max_epochs=config['training']['max_epochs'],
        class_names=config.get('class_names'),
        he_retention_loss_weight=config['model']['he_retention_loss_weight'],
        ihc_retention_loss_weight=config['model']['ihc_retention_loss_weight'],
        he_ce_loss_weight=config['model']['he_ce_loss_weight'],
        ihc_ce_loss_weight=config['model']['ihc_ce_loss_weight']
    )

    # Print model info (only for first fold to avoid repetition)
    if fold == 0:
        print(f"Trainable parameters: {model.get_num_trainable_params():,}")
        print(f"Total parameters: {model.get_total_params():,}")

    # Setup callbacks and logger
    callbacks = setup_callbacks(config)
    logger = setup_logger(config, experiment_name, output_dir)

    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator=config['training']['accelerator'],
        devices=config['training']['devices'],
        precision=config['training']['precision'],
        gradient_clip_val=config['training']['gradient_clip_val'],
        accumulate_grad_batches=config['training']['accumulate_grad_batches'],
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=config['logging']['log_every_n_steps'],
        deterministic=True,  # For reproducibility
        enable_progress_bar=True
    )

    # Train the model
    trainer.fit(model, data_module)

    # Test the model
    test_results = trainer.test(model, data_module, ckpt_path='best')

    return test_results[0], trainer.checkpoint_callback.best_model_path



def main():
    parser = argparse.ArgumentParser(description='Train MIL models with Cross-Validation on histopathology data')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Directory to save outputs')
    parser.add_argument('--model_name', type=str, help='Override model name from config')
    parser.add_argument('--fold', type=int, default=0, help='index of fold to run')
    parser.add_argument('--tag', type=str, default='', help='experiment tag for logging')
    parser.add_argument('--gpu', type=int, help='GPU device to use')

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override GPU if specified
    if args.gpu is not None:
        config['training']['devices'] = [args.gpu]

    # Set random seed for reproducibility
    pl.seed_everything(config['data']['random_seed'])

    # Determine which folds to run
    fold = args.fold

    # Create output directory
    args.output_dir = args.output_dir + f'_fold{fold}'
    os.makedirs(args.output_dir, exist_ok=True)

    # Copy config to output directory
    config_backup_path = os.path.join(args.output_dir, 'config.yaml')
    shutil.copy2(args.config, config_backup_path)
    print(f"Config file saved to: {config_backup_path}")

    # Determine model name
    model_name = args.model_name or config['model']['model_name']

    # Run experiment(s)
    # Single fold experiment
    test_results, best_ckpt = train_single_fold(config, model_name, fold, args.tag, args.output_dir)

    print(f"\n{'='*60}")
    print(f"SINGLE FOLD EXPERIMENT COMPLETED")
    print(f"Fold {fold} Results:")
    for key, value in test_results.items():
        print(f"  {key}: {value:.4f}")
    print(f"Best checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()
