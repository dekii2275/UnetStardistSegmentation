import torch
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader, random_split

from config import Config
from dataset import DSB2018Dataset
from transform import get_transforms
from modelUnet import UNet 

class ModelEvaluator:
    def __init__(self, model, dataloader, device):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        # Các ngưỡng IoU từ 0.5 đến 0.95 (step 0.05) theo chuẩn DSB2018
        self.thresholds = np.arange(0.5, 1.0, 0.05) 
        
    def _compute_iou_batch(self, y_pred, y_true):
        """
        Tính IoU và Precision cho từng ảnh trong batch tại nhiều ngưỡng IoU.
        """
        # 1. Post-processing: Tách instance bằng Connected Components
        num_true, labels_true = cv2.connectedComponents(y_true.astype(np.uint8))
        num_pred, labels_pred = cv2.connectedComponents(y_pred.astype(np.uint8))
        
        # Xử lý các trường hợp biên (edge cases)
        if num_true == 1: 
            # Nếu không có object thật nào
            # Nếu cũng không có pred nào -> Đúng (1.0), ngược lại -> Sai (0.0)
            return [1.0 if num_pred == 1 else 0.0] * len(self.thresholds)
            
        if num_pred == 1: 
            # Có object thật mà không dự đoán được cái nào -> Sai hết (0.0)
            return [0.0] * len(self.thresholds)
        
        # 2. Tính Ma trận IoU (Intersection over Union)
        # labels_true và labels_pred chứa các ID: 0 (nền), 1, 2, 3... (tế bào)
        
        # Tạo IoU matrix [số tế bào thật, số tế bào dự đoán]
        # (Trừ 1 vì không tính nền)
        iou_matrix = np.zeros((num_true-1, num_pred-1))
        
        for i in range(1, num_true):
            true_mask = (labels_true == i)
            true_area = np.sum(true_mask)
            
            # Tìm các label dự đoán chồng lấn với tế bào thật này
            intersect_labels = labels_pred[true_mask]
            intersect_labels = intersect_labels[intersect_labels > 0] # Bỏ nền
            
            if len(intersect_labels) == 0: continue
            
            # Tính IoU cho từng cặp chồng lấn
            pred_ids, counts = np.unique(intersect_labels, return_counts=True)
            for pid, overlap_area in zip(pred_ids, counts):
                pred_area = np.sum(labels_pred == pid)
                union = true_area + pred_area - overlap_area
                iou = overlap_area / union
                iou_matrix[i-1, pid-1] = iou
                
        # 3. Tính Precision tại các ngưỡng IoU
        precisions = []
        for t in self.thresholds:
            # Matches: Những cặp có IoU > ngưỡng t
            matches = iou_matrix > t
            
            # Đếm số lượng True Positives (TP)
            # Mỗi object thật chỉ được match tối đa 1 object giả (Lấy IoU cao nhất)
            tp = 0
            used_preds = set()
            
            for i in range(iou_matrix.shape[0]):
                if iou_matrix.shape[1] > 0:
                    # Tìm match tốt nhất cho object thật i
                    best_match_idx = np.argmax(iou_matrix[i])
                    max_iou = iou_matrix[i, best_match_idx]
                    
                    if max_iou > t and best_match_idx not in used_preds:
                        tp += 1
                        used_preds.add(best_match_idx)
            
            # Công thức Precision DSB: TP / (TP + FP + FN)
            fp = (num_pred - 1) - tp
            fn = (num_true - 1) - tp
            
            score = tp / (tp + fp + fn + 1e-7)
            precisions.append(score)
            
        return precisions

    def run(self):
        self.model.eval()
        avg_precisions = np.zeros(len(self.thresholds))
        count = 0
        
        print(f"🔍 Đang đánh giá mAP trên {len(self.dataloader)} batch...")
        
        with torch.no_grad():
            for batch in tqdm(self.dataloader):
                images, masks = batch
                images = images.to(self.device, dtype=torch.float32)
                
                # Predict
                outputs = self.model(images)
                preds = torch.sigmoid(outputs)
                
                # Chuyển về numpy để xử lý Connected Components
                preds = preds.cpu().numpy()
                masks = masks.numpy()
                
                # Loop qua từng ảnh trong batch
                for i in range(preds.shape[0]):
                    # Binarize (Ngưỡng xác suất 0.5 để tạo mask nhị phân)
                    pred_mask = (preds[i, 0] > 0.5).astype(np.uint8)
                    true_mask = (masks[i, 0] > 0.5).astype(np.uint8)
                    
                    # Tính Precision tại các ngưỡng IoU
                    scores = self._compute_iou_batch(pred_mask, true_mask)
                    avg_precisions += np.array(scores)
                    count += 1
        
        # Trung bình hóa trên toàn bộ tập dữ liệu
        if count > 0:
            avg_precisions /= count
        
        mAP = np.mean(avg_precisions)
        return self.thresholds, avg_precisions, mAP

    def plot(self, thresholds, precisions, mAP):
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, precisions, 'o-', color='crimson', linewidth=2, label='Precision')
        plt.axhline(y=mAP, color='navy', linestyle='--', label=f'Mean AP = {mAP:.4f}')
        
        plt.title('Precision at IoU Thresholds', fontsize=14, fontweight='bold')
        plt.xlabel('IoU Threshold', fontsize=12)
        plt.ylabel('Average Precision', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.xticks(thresholds)
        plt.ylim(-0.05, 1.05)
        
        # In giá trị lên điểm
        for x, y in zip(thresholds, precisions):
            plt.text(x, y+0.02, f'{y:.2f}', ha='center', fontsize=9)
            
        plt.tight_layout()
        plt.savefig('evaluation_result.png') # Lưu ảnh thay vì show
        print(f"📊 Đã lưu biểu đồ kết quả vào 'evaluation_result.png'")
        print(f"\n🏆 KẾT QUẢ CUỐI CÙNG: mAP = {mAP:.4f}")
        
if __name__ == "__main__":
    # 1. Cấu hình & Thiết bị
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Device: {device}")
    
    cfg = Config()
    
    # 2. Load Model
    # Lưu ý: Sửa 'n_channels', 'n_classes' cho khớp với lúc train
    model = UNet(n_channels=3, n_classes=1).to(device)
    
    checkpoint_path = 'best_unet_checkpoint.pth'
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"✅ Đã load checkpoint: {checkpoint_path}")
    except FileNotFoundError:
        print(f"⚠️ Không tìm thấy file {checkpoint_path}. Đang chạy với model ngẫu nhiên để test code...")
    
    # 3. Load Validation Data
    # Chúng ta load lại dataset và tách ra phần validation giống như lúc train
    # để đảm bảo đánh giá trên dữ liệu model chưa từng học.
    full_dataset = DSB2018Dataset(
        root_dir=cfg.TRAIN_DIR,
        transform=get_transforms(cfg, stage='test') # Dùng 'test' để chỉ resize, không augment
    )
    
    # Giả sử tách 90/10 như file train.py
    n_val = int(len(full_dataset) * 0.1)
    n_train = len(full_dataset) - n_val
    _, val_set = random_split(full_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(cfg.SEED))
    
    # Tạo loader
    val_loader = DataLoader(val_set, batch_size=4, shuffle=False, num_workers=0)
    print(f"📦 Dữ liệu đánh giá: {len(val_set)} ảnh")

    # 4. Chạy Đánh giá
    evaluator = ModelEvaluator(model, val_loader, device)
    thresholds, precisions, mAP = evaluator.run()
    
    # 5. Vẽ & Lưu kết quả
    evaluator.plot(thresholds, precisions, mAP)