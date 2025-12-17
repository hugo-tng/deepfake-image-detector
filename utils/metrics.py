import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report, roc_auc_score
)
from typing import Tuple 

class MetricsTracker:
    """Track and compute metrics"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        # Lưu trữ các tensor con trên thiết bị gốc (GPU)
        self.predictions = []
        self.targets = []
        self.losses = []
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor, loss: float):
        """
        Update with batch results.
        'preds' và 'targets' là các tensor trên GPU.
        'loss' là một giá trị float (từ loss.item())
        """
        # Detach tensor để tránh lưu toàn bộ computation graph
        self.predictions.append(preds.detach())
        self.targets.append(targets.detach())
        self.losses.append(loss)
    
    def _get_preds_targets_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Helper: Nối (concat) tất cả tensor và chuyển về CPU/Numpy MỘT LẦN"""
        if not self.predictions:
            return np.array([]), np.array([])
        
        # Nối tất cả các batch lại với nhau
        all_preds = torch.cat(self.predictions, dim=0)
        all_targets = torch.cat(self.targets, dim=0)
        
        # Chuyển về CPU/Numpy
        preds_np = all_preds.cpu().numpy()
        targets_np = all_targets.cpu().numpy()
        
        return preds_np, targets_np

    def compute_metrics(self) -> dict:
        """Compute all metrics (chỉ gọi ở cuối epoch)"""
        
        preds_np, targets_np = self._get_preds_targets_numpy()
        
        if len(preds_np) == 0:
            # Trả về giá trị 0 nếu không có dữ liệu
            return {
                'loss': 0.0, 'accuracy': 0.0, 'precision': 0.0,
                'recall': 0.0, 'f1': 0.0
            }
            
        metrics = {
            'loss': np.mean(self.losses),
            'accuracy': accuracy_score(targets_np, preds_np),
            'precision': precision_score(targets_np, preds_np, zero_division=0),
            'recall': recall_score(targets_np, preds_np, zero_division=0),
            'f1': f1_score(targets_np, preds_np, zero_division=0)
        }
        
        return metrics
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix (chỉ gọi ở cuối epoch)"""
        preds_np, targets_np = self._get_preds_targets_numpy()
        if len(preds_np) == 0:
            return np.zeros((2, 2))
        return confusion_matrix(targets_np, preds_np)
    
    def get_classification_report(self) -> str:
        """Get detailed classification report (chỉ gọi ở cuối epoch)"""
        preds_np, targets_np = self._get_preds_targets_numpy()
        if len(preds_np) == 0:
            return "No data to report."
        return classification_report(
            targets_np, 
            preds_np,
            target_names=['Real', 'Fake'],
            zero_division=0
        )