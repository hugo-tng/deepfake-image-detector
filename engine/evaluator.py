import json
import torch
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, classification_report, confusion_matrix
)

from utils.visualization import *

@torch.no_grad()
def evaluate_test_set(model, test_loader, device, output_dir):
    """Chạy đánh giá trên tập test"""
    print("\n" + "="*50)
    print("BẮT ĐẦU ĐÁNH GIÁ TRÊN TẬP TEST")
    print("="*50)
    
    all_preds = []
    all_labels = []
    all_probs = [] # Lưu xác suất lớp 1 (Fake) để vẽ ROC
    
    # 1. Vòng lặp dự đoán
    for images, labels in tqdm(test_loader, desc="Testing"):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward
        outputs = model(images)
        
        # Lấy xác suất (Softmax)
        probs = torch.softmax(outputs, dim=1)
        
        # Lấy nhãn dự đoán
        _, preds = torch.max(outputs, 1)
        
        # Gom kết quả
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy()) # Lấy cột index 1 (Fake)
    
    # 2. Tính toán Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    print(f"\n📊 KẾT QUẢ ĐÁNH GIÁ:")
    print(f"   Accuracy : {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall   : {recall:.4f}")
    print(f"   F1 Score : {f1:.4f}")
    
    print("\n📋 Chi tiết theo lớp:")
    print(classification_report(all_labels, all_preds, target_names=['Real', 'Fake']))

    # 3. Trực quan hóa & Lưu
    # Tạo thư mục evaluation riêng bên trong thư mục experiment
    eval_dir = os.path.join(output_dir, 'evaluation_results')
    os.makedirs(eval_dir, exist_ok=True)
    
    # A. Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, classes=['Real', 'Fake'], save_dir=eval_dir)
    
    # B. ROC Curve & AUC
    auc_score = plot_roc_curve(all_labels, all_probs, save_dir=eval_dir)
    print(f"=== ROC AUC Score: {auc_score:.4f}")
    
    # C. Lưu kết quả dạng JSON
    results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'auc_score': float(auc_score),
        'confusion_matrix': cm.tolist()
    }
    
    json_path = os.path.join(eval_dir, 'test_metrics.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\nĐã lưu toàn bộ kết quả đánh giá tại: {eval_dir}")