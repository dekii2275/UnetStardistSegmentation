import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast # Mixed Precision
from tqdm import tqdm
import numpy as np

# Import modules
from config import Config
from dataset_stardist import StarDistDataset # <--- Đảm bảo bạn đã sửa tên file đúng
from transform import get_transforms
from model_stardist import StarDist
from utils import seed_everything

def stardist_loss(pred_prob_logits, pred_dist, target_prob, target_dist, grid=(1,1)):
    """
    Hàm Loss tùy chỉnh cho StarDist
    grid: dùng để cân bằng trọng số (mặc định 1)
    """
    # 1. Probability Loss (BCEWithLogits)
    # target_prob shape: (B, 1, H, W)
    loss_prob = F.binary_cross_entropy_with_logits(pred_prob_logits, target_prob)

    # 2. Distance Loss (L1 Loss)
    # Chỉ tính loss khoảng cách tại những điểm LÀ tế bào (target_prob > 0)
    # mask shape: (B, 1, H, W) -> broadcast sang (B, 32, H, W)
    mask = target_prob > 0
    
    # Lọc những pixel thuộc object
    # Nếu mask rỗng (ảnh đen xì) thì loss_dist = 0
    if mask.sum() > 0:
        # Expand mask để khớp số tia (n_rays)
        mask_dist = mask.expand_as(pred_dist)
        
        # Tính L1 loss trên vùng mask
        loss_dist = F.l1_loss(pred_dist[mask_dist], target_dist[mask_dist])
    else:
        loss_dist = torch.tensor(0.0, device=pred_dist.device)

    # Tổng hợp loss (có thể thêm trọng số alpha nếu cần)
    # Thường distance loss lớn hơn prob loss nhiều, nên ta cộng trực tiếp
    total_loss = loss_prob + loss_dist
    
    return total_loss, loss_prob.item(), loss_dist.item()

def train_stardist():
    # 1. Setup
    cfg = Config()
    seed_everything(cfg.SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Device: {device} | Batch Size: {cfg.BATCH_SIZE}")

    # 2. Data
    # Lưu ý: n_rays=32 phải khớp với model
    full_dataset = StarDistDataset(
        root_dir=cfg.TRAIN_DIR,
        transform=get_transforms(cfg, stage='train'),
        n_rays=32 
    )

    n_val = int(len(full_dataset) * 0.1)
    n_train = len(full_dataset) - n_val
    train_set, val_set = random_split(full_dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=True)
    
    print(f"📦 Data: {n_train} Train | {n_val} Val (StarDist Mode)")

    # 3. Model
    model = StarDist(n_channels=3, n_rays=32).to(device)

    # 4. Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4) # Adam thường tốt cho StarDist
    scaler = GradScaler() # Dùng Mixed Precision để tiết kiệm VRAM

    EPOCHS = 100 # StarDist cần train lâu hơn U-Net chút
    min_loss = float('inf')

    print("\n🏁 START TRAINING STARDIST...")
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        p_loss_total = 0
        d_loss_total = 0
        
        with tqdm(total=len(train_set), desc=f'Epoch {epoch+1}/{EPOCHS}', unit='img') as pbar:
            for batch in train_loader:
                # Unpack 3 thành phần: Ảnh, Xác suất, Khoảng cách
                images, target_probs, target_dists = batch
                
                images = images.to(device, dtype=torch.float32)
                target_probs = target_probs.to(device, dtype=torch.float32)
                target_dists = target_dists.to(device, dtype=torch.float32)

                optimizer.zero_grad()

                # Mixed Precision Forward
                with autocast():
                    pred_logits, pred_dists = model(images)
                    loss, l_prob, l_dist = stardist_loss(pred_logits, pred_dists, target_probs, target_dists)

                # Backward
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # Log
                epoch_loss += loss.item()
                p_loss_total += l_prob
                d_loss_total += l_dist
                
                pbar.set_postfix({'L_Total': loss.item(), 'L_Prob': l_prob, 'L_Dist': l_dist})
                pbar.update(images.shape[0])

        # --- VALIDATION (Tính Loss trung bình) ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                images, target_probs, target_dists = batch
                images = images.to(device, dtype=torch.float32)
                target_probs = target_probs.to(device, dtype=torch.float32)
                target_dists = target_dists.to(device, dtype=torch.float32)

                pred_logits, pred_dists = model(images)
                loss, _, _ = stardist_loss(pred_logits, pred_dists, target_probs, target_dists)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"📊 Epoch {epoch+1}: Train Loss = {epoch_loss/len(train_loader):.4f} | Val Loss = {avg_val_loss:.4f}")

        # Save Best
        if avg_val_loss < min_loss:
            min_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_stardist_checkpoint.pth')
            print(f"💾 Saved Best StarDist Model (Loss: {min_loss:.4f})")

if __name__ == '__main__':
    train_stardist()