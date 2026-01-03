import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import json

from utils.metrics import MetricsTracker
from utils.config import TrainingConfig, GlobalConfig

class Trainer:
    """Main training class"""
    
    def __init__(self, model, train_loader: DataLoader, val_loader: DataLoader, training_config: TrainingConfig):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = training_config

        self.device = GlobalConfig.DEVICE
        self.model.to(self.device)

        self.freeze_epochs = getattr(self.cfg, "FREEZE_EPOCHS", 0)
        self._last_freeze_state = self.model.spatial_frozen
        # KHỞI TẠO OPTIMIZER 
        self.optimizer = self._build_optimizer()  
        # Scheduler
        self.scheduler = self._build_scheduler()

        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.cfg.LABEL_SMOOTHING)

        self.scaler = GradScaler('cuda') if self.cfg.USE_AMP else None
        
        self.train_metrics = MetricsTracker()
        self.val_metrics = MetricsTracker()
        
        self.history = {
            'train_loss': [], 'train_acc': [], 'train_f1': [],
            'val_loss': [], 'val_acc': [], 'val_precision': [], 
            'val_recall': [], 'val_f1': [],
            'learning_rates': []
        }
        
        self.best_val_f1 = 0.0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        self.cfg.create_directories()

    def _build_optimizer(self):
        param_groups = []

        # Backbone (EfficientNet)
        if self.model.spatial_branch is not None:
            backbone_params = [
                p for p in self.model.spatial_branch.backbone.parameters()
                if p.requires_grad
            ]
            if backbone_params:
                param_groups.append({
                    "params": backbone_params,
                    "lr": self.cfg.LEARNING_RATE * 0.1
                })

            # Projection 
            proj_params = list(
                self.model.spatial_branch.projection.parameters()
            )
            param_groups.append({
                "params": proj_params,
                "lr": self.cfg.LEARNING_RATE
            })

        # Frequency branch
        if self.model.frequency_branch is not None:
            param_groups.append({
                "params": self.model.frequency_branch.parameters(),
                "lr": self.cfg.LEARNING_RATE
            })

        # Fusion + Classifier
        other_params = []
        if self.model.fusion is not None:
            other_params += list(self.model.fusion.parameters())
        other_params += list(self.model.classifier.parameters())

        param_groups.append({
            "params": other_params,
            "lr": self.cfg.LEARNING_RATE
        })

        return optim.AdamW(
            param_groups,
            weight_decay=self.cfg.WEIGHT_DECAY
        )

    
    def _build_scheduler(self):
        sched = self.cfg.SCHEDULER_TYPE.lower()

        if sched == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.cfg.T_MAX,
                eta_min=1e-6
            )

        elif sched in ["plateau", "reducelronplateau"]:
            return ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=3,
                verbose=True
            )
        else:
            raise ValueError(f"Unknown scheduler: {sched}")
        
    def _handle_freeze_unfreeze(self, epoch):
        if self.freeze_epochs <= 0:
            return

        if epoch < self.freeze_epochs:
            if not self.model.spatial_frozen:
                print(f"[INFO] Freezing backbone at epoch {epoch}")
                self.model.freeze_spatial_backbone(True)
        else:
            if self.model.spatial_frozen:
                print(f"[INFO] Unfreezing backbone at epoch {epoch}")
                self.model.freeze_spatial_backbone(False)

                if isinstance(self.scheduler, CosineAnnealingLR):
                    self.scheduler.last_epoch = -1


    def train_epoch(self, epoch: int):
        self._handle_freeze_unfreeze(epoch)

        self.model.train()
        self.train_metrics.reset()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.cfg.NUM_EPOCHS}")

        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)

            if self.cfg.USE_AMP:
                with autocast('cuda'):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad], 1.0
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()

            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            _, preds = torch.max(outputs, 1)
            self.train_metrics.update(preds, labels, loss.item())

            if batch_idx % 10 == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{self.optimizer.param_groups[-1]["lr"]:.6f}'
                })

        return self.train_metrics.compute_metrics()
    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        self.val_metrics.reset()
        for images, labels in tqdm(self.val_loader, desc="Validating"):
            images, labels = images.to(self.device), labels.to(self.device)
            if self.cfg.USE_AMP:
                with autocast('cuda'):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            self.val_metrics.update(preds, labels, loss.item())
        return self.val_metrics.compute_metrics()
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        checkpoint_dir = self.cfg.CHECKPOINT_DIR
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.cfg.MODEL_CONFIG,
            'history': self.history
        }

        torch.save(checkpoint, os.path.join(self.cfg.CHECKPOINT_DIR, "last_model.pth"))
        
        if not getattr(self.cfg, 'SAVE_BEST_ONLY', False):
            torch.save(checkpoint, os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth"))
        
        if is_best:
            torch.save(checkpoint, os.path.join(checkpoint_dir, "best_model.pth"))
            print(f"== Best model saved.")

    def _update_history(self, train_metrics: dict, val_metrics: dict):
        """Update training history"""
        self.history['train_loss'].append(train_metrics['loss'])
        self.history['train_acc'].append(train_metrics['accuracy'])
        self.history['train_f1'].append(train_metrics['f1'])
        
        self.history['val_loss'].append(val_metrics['loss'])
        self.history['val_acc'].append(val_metrics['accuracy'])
        self.history['val_precision'].append(val_metrics['precision'])
        self.history['val_recall'].append(val_metrics['recall'])
        self.history['val_f1'].append(val_metrics['f1'])
        
        self.history['learning_rates'].append([
            pg['lr'] for pg in self.optimizer.param_groups
        ])


    def _print_epoch_summary(self, epoch: int, train_metrics: dict, val_metrics: dict):
        """Print epoch summary with comprehensive metrics"""
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{self.cfg.NUM_EPOCHS} Summary")
        print(f"{'='*70}")
        print(f"Train | Loss: {train_metrics['loss']:.4f} | "
              f"Acc: {train_metrics['accuracy']:.4f} | "
              f"F1: {train_metrics['f1']:.4f}")
        print(f"Val   | Loss: {val_metrics['loss']:.4f} | "
              f"Acc: {val_metrics['accuracy']:.4f}")
        print(f"      | Precision: {val_metrics['precision']:.4f} | "
              f"Recall: {val_metrics['recall']:.4f} | "
              f"F1: {val_metrics['f1']:.4f}")
        print(f"LR    | {self.optimizer.param_groups[0]['lr']:.6f}")

    def train(self):
        print(f"\nSTARTING TRAINING | Mode: {self.cfg.MODE}")
        print("="*70)
        
        for epoch in range(self.cfg.NUM_EPOCHS):
            print(f"\nEpoch {epoch+1}/{self.cfg.NUM_EPOCHS}")
            
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()
            
            # Scheduler update
            if isinstance(self.scheduler, CosineAnnealingLR):
                self.scheduler.step()
            else:
                self.scheduler.step(val_metrics['loss'])
            
            # History Update
            self._update_history(train_metrics, val_metrics)

            self._print_epoch_summary(epoch, train_metrics, val_metrics)
            
            # Early stopping & best model based on F1
            is_best = val_metrics['f1'] > self.best_val_f1
            if is_best:
                self.best_val_f1 = val_metrics['f1']
                self.epochs_without_improvement = 0
                print(f"   New Best Val F1: {self.best_val_f1:.4f}")
            else:
                self.epochs_without_improvement += 1
                print(f"   Epochs without F1 improvement: {self.epochs_without_improvement}/{self.cfg.EARLY_STOPPING_PATIENCE}")

            self.save_checkpoint(epoch, val_metrics, is_best=is_best)

            if self.epochs_without_improvement >= self.cfg.EARLY_STOPPING_PATIENCE:
                print(f"\n [INFO] Early stopping triggered based on F1-score.")
                break

        # Save training history
        log_dir = getattr(self.cfg, 'LOG_DIR', self.cfg.CHECKPOINT_DIR)
        history_path = os.path.join(log_dir, 'training_history.json')
        
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        
        print(f"\nTraining Completed. Best F1: {self.best_val_f1:.4f}")
        print(f"History saved to: {history_path}")