import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score, ConfusionMatrix, AUROC
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from losses import CMDLoss


class ModelInterface(pl.LightningModule):
    """
    PyTorch Lightning module for end-to-end MIL model.
    """
    
    def __init__(
        self,
        num_classes: int = 9,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        scheduler: str = "cosine",  # "cosine", "step", or "none"
        max_epochs: int = 100,
        class_names: list = None,
        he_retention_loss_weight=0.1,
        ihc_retention_loss_weight=0.1,
        he_ce_loss_weight=1.0,
        ihc_ce_loss_weight=1.0,
    ):
        """
        Args:
            num_classes: Number of output classes
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            scheduler: Type of learning rate scheduler
            max_epochs: Maximum number of epochs (for cosine scheduler)
            class_names: List of class names for logging

        """
        super(ModelInterface, self).__init__()

        
        # Save hyperparameters
        self.save_hyperparameters()
        
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.max_epochs = max_epochs
        self.loss = CMDLoss(he_retention_loss_weight, ihc_retention_loss_weight, he_ce_loss_weight, ihc_ce_loss_weight)
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        
        # Build the model using your model builder
        self.model = self._build_mymodel(num_classes)
        
        # Initialize metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes, average='weighted')
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes, average='weighted')
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes, average='weighted')
        
        self.train_f1 = F1Score(task="multiclass", num_classes=num_classes, average='weighted')
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average='weighted')
        self.test_f1 = F1Score(task="multiclass", num_classes=num_classes, average='weighted')

        self.train_auc = AUROC(task="multiclass", num_classes=num_classes, average='macro')
        self.val_auc = AUROC(task="multiclass", num_classes=num_classes, average='macro')
        self.test_auc = AUROC(task="multiclass", num_classes=num_classes, average='macro')

        self.train_macro_f1 = F1Score(task="multiclass", num_classes=num_classes, average='macro')
        self.val_macro_f1 = F1Score(task="multiclass", num_classes=num_classes, average='macro')
        self.test_macro_f1 = F1Score(task="multiclass", num_classes=num_classes, average='macro')

        # For confusion matrix
        self.test_cm = ConfusionMatrix(task="multiclass", num_classes=num_classes)
        
        # Store test predictions for analysis
        self.test_step_outputs = []
    
    def _build_mymodel(self, num_classes):
        """Build linear probe using your model builder"""
        from models.mymodel import CMD
        

        # Build the linear probe - ONLY one linear layer
        model = CMD(wsi_embed_dim=1536, embed_dim=512, num_classes=num_classes)

        return model
        
    def forward(self, he_emb, ihc_emb):
        return self.model(he_emb, ihc_emb)
    
    def training_step(self, batch, batch_idx):
        (he_emb, ihc_emb), labels = batch
        (
            he_retention_emb,
            he_retention_target,
            he_mask,
            he_results_dict,
            ihc_retention_emb,
            ihc_retention_target,
            ihc_mask,
            ihc_results_dict,
        ) = self(he_emb, ihc_emb)

        (
            loss,
            he_retention_loss,
            ihc_retention_loss,
            he_ce_loss,
            ihc_ce_loss,
        ) = self.loss(
            he_retention_emb,
            he_retention_target,
            he_mask,
            he_results_dict,
            ihc_retention_emb,
            ihc_retention_target,
            ihc_mask,
            ihc_results_dict,
            labels,
        )
        
        # Calculate metrics
        preds = he_results_dict["preds"]
        probs = he_results_dict["probs"]
        
        self.train_acc(preds, labels)
        self.train_macro_f1(preds, labels)
        self.train_auc(probs, labels)  # AUC needs probabilities
        
        # Log metrics
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/he_retention_loss', he_retention_loss, on_step=True, on_epoch=True)
        self.log('train/ihc_retention_loss', ihc_retention_loss, on_step=True, on_epoch=True)
        self.log('train/he_ce_loss', he_ce_loss, on_step=True, on_epoch=True)
        self.log('train/ihc_ce_loss', ihc_ce_loss, on_step=True, on_epoch=True)
        self.log('train/acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/macf1', self.train_macro_f1, on_step=False, on_epoch=True)
        self.log('train/macauc', self.train_auc, on_step=False, on_epoch=True)
        
        return loss

    def forward_validate(self, x):
        return self.model.foward_validate(x)
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        he_results_dict = self.forward_validate(images)
        loss = F.cross_entropy(he_results_dict["logits"], labels)
        
        # Calculate metrics
        preds = he_results_dict["preds"]
        probs = he_results_dict["probs"]
        
        self.val_acc(preds, labels)
        self.val_f1(preds, labels)
        self.val_auc(probs, labels)  # AUC needs probabilities
        
        # Log metrics
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/macf1', self.val_macro_f1, on_step=False, on_epoch=True)
        self.log('val/macauc', self.val_auc, on_step=False, on_epoch=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        images, labels = batch
        he_results_dict = self.forward_validate(images)
        loss = F.cross_entropy(he_results_dict["logits"], labels)
        
        # Calculate metrics
        preds = he_results_dict["preds"]
        probs = he_results_dict["probs"]
        
        self.test_acc(preds, labels)
        self.test_macro_f1(preds, labels)
        self.test_f1(preds, labels)
        self.test_auc(probs, labels)  # AUC needs probabilities
        self.test_cm(preds, labels)
        
        # Store for later analysis
        self.test_step_outputs.append({
            'preds': preds.cpu(),
            'labels': labels.cpu(),
            'logits': he_results_dict["logits"].cpu(),
            'probs': probs.cpu()
        })
        
        # Log metrics
        self.log('test/loss', loss, on_step=False, on_epoch=True)
        self.log('test/acc', self.test_acc, on_step=False, on_epoch=True)
        self.log('test/macf1', self.test_macro_f1, on_step=False, on_epoch=True)
        self.log('test/wf1', self.test_f1, on_step=False, on_epoch=True)
        self.log('test/macauc', self.test_auc, on_step=False, on_epoch=True)
        
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Prediction step for inference"""
        images, labels = batch
        
        # Get model predictions - your model returns (attention_weights, logits)
        he_results_dict = self.forward_validate(images)
        preds = he_results_dict["preds"]
        probs = he_results_dict["probs"]
        
        return {
            'probs': probs,
            'preds': preds,
            'labels': labels
        }
    
    def on_test_epoch_end(self):
        """Generate confusion matrix and per-class metrics at end of testing"""
        # Plot confusion matrix
        cm = self.test_cm.compute().cpu().numpy()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title(f'Confusion Matrix - {self.model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Log the plot to tensorboard/wandb if available
        if self.logger:
            import wandb
            self.logger.experiment.log({
                'test/confusion_matrix': wandb.Image(plt.gcf())
            })
        plt.close()
        
        # Calculate per-class accuracy
        per_class_acc = cm.diagonal() / cm.sum(axis=1)
        for i, acc in enumerate(per_class_acc):
            self.log(f'test/acc_class_{self.class_names[i]}', acc)
        
        # Clear test outputs
        self.test_step_outputs.clear()
    
    def configure_optimizers(self):
        # Optimize entire model
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        if self.scheduler == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer, 
                T_max=self.max_epochs,
                eta_min=self.learning_rate * 0.01
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch'
                }
            }
        elif self.scheduler == "step":
            scheduler = StepLR(
                optimizer,
                step_size=30,
                gamma=0.1
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch'
                }
            }
        else:
            return optimizer
    
    
    def get_num_trainable_params(self):
        """Get number of trainable parameters (should be just the linear layer)"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_params(self):
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())