import os
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import roc_curve, auc

def plot_training_history(history_dict: dict, save_dir: str, show: bool = True):
    """
    Vẽ biểu đồ Loss, Accuracy và Learning Rate từ lịch sử huấn luyện.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Tạo figure với lưới 2x2
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    epochs = range(1, len(history_dict['train_loss']) + 1)
    
    # 1. Plot Loss
    axes[0, 0].plot(epochs, history_dict['train_loss'], label='Train Loss', marker='.', color='tab:blue')
    axes[0, 0].plot(epochs, history_dict['val_loss'], label='Val Loss', marker='.', color='tab:orange')
    axes[0, 0].set_title('Training & Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Plot Accuracy
    axes[0, 1].plot(epochs, history_dict['train_acc'], label='Train Acc', marker='.', color='tab:green')
    axes[0, 1].plot(epochs, history_dict['val_acc'], label='Val Acc', marker='.', color='tab:red')
    axes[0, 1].set_title('Training & Validation Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Plot Learning Rate(s)
    lrs = history_dict['learning_rates']

    # Backbone = group 0
    backbone_lr = [epoch_lr[0] for epoch_lr in lrs]

    # Other groups (assume same LR): take group 1
    other_lr = [epoch_lr[1] for epoch_lr in lrs]

    axes[1, 0].plot(
        epochs,
        backbone_lr,
        marker='x',
        label='Backbone LR'
    )

    axes[1, 0].plot(
        epochs,
        other_lr,
        marker='o',
        label='Projection + FFT + Fusion LR'
    )

    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_yscale('log')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Text Info (Tổng kết)
    axes[1, 1].axis('off')
    best_acc = max(history_dict['val_acc']) if history_dict['val_acc'] else 0
    last_acc = history_dict['val_acc'][-1] if history_dict['val_acc'] else 0
    min_loss = min(history_dict['val_loss']) if history_dict['val_loss'] else 0
    
    info_text = (
        f"TRAINING RESULT SUMMARY\n"
        f"-----------------------\n"
        f"Best Val Acc : {best_acc:.4f}\n"
        f"Last Val Acc : {last_acc:.4f}\n"
        f"Min Val Loss : {min_loss:.4f}\n"
        f"Total Epochs : {len(history_dict['train_loss'])}"
    )
    axes[1, 1].text(0.5, 0.5, info_text, ha='center', va='center', fontsize=14, 
                    bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", alpha=0.8))

    plt.tight_layout()
    
    # Lưu ảnh
    save_path = os.path.join(save_dir, 'training_history.png')
    plt.savefig(save_path, dpi=150)
    print(f"📈 Đã lưu biểu đồ training tại: {save_path}")
    
    # Hiển thị ảnh
    if show:
        plt.show()
    
    plt.close()

def plot_validation_metrics(history_dict, save_dir, show=True):
    """
    Vẽ biểu đồ so sánh chi tiết các chỉ số Validation:
    Accuracy, Precision, Recall, F1-Score trên cùng một đồ thị.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    metrics = ['val_acc', 'val_precision', 'val_recall', 'val_f1']
    labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6'] # Xanh lá, Xanh dương, Đỏ, Tím
    markers = ['o', 's', '^', 'D']
    
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(history_dict['val_acc']) + 1)
    
    has_data = False
    for i, metric in enumerate(metrics):
        # Kiểm tra xem metric có tồn tại và có dữ liệu không
        if metric in history_dict and len(history_dict[metric]) > 0:
            plt.plot(epochs, history_dict[metric], label=labels[i], 
                     color=colors[i], marker=markers[i], linewidth=2, markersize=5)
            has_data = True
            
    if not has_data:
        print("⚠️ Không tìm thấy dữ liệu metrics chi tiết để vẽ.")
        plt.close()
        return

    plt.title('Validation Metrics Analysis over Epochs', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.ylim(0.0, 1.05) # Giới hạn trục Y từ 0 đến 1
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(loc='lower right', fontsize=10)
    
    plt.tight_layout()
    
    # Lưu ảnh
    save_path = os.path.join(save_dir, 'validation_metrics.png')
    plt.savefig(save_path, dpi=150)
    print(f"📊 Đã lưu biểu đồ Validation Metrics tại: {save_path}")
    
    # Hiển thị ảnh
    if show:
        plt.show()
        
    plt.close()

def plot_confusion_matrix(cm, classes, save_dir, filename='confusion_matrix.png', show=True):
    """Vẽ và lưu Confusion Matrix"""
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    # Hiển thị số lượng
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=14, fontweight='bold')

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Lưu ảnh
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=150)
    print(f"🔢 Đã lưu Confusion Matrix tại: {save_path}")
    
    # Hiển thị ảnh
    if show:
        plt.show()
        
    plt.close()

def plot_roc_curve(y_true, y_probs, save_dir, show=True):
    """Vẽ đường cong ROC"""
    os.makedirs(save_dir, exist_ok=True)
    
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    # Lưu ảnh
    save_path = os.path.join(save_dir, 'roc_curve.png')
    plt.savefig(save_path, dpi=150)
    print(f"📉 Đã lưu ROC Curve tại: {save_path}")
    
    # Hiển thị ảnh
    if show:
        plt.show()
        
    plt.close()
    
    return roc_auc