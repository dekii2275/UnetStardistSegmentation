import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os

# Import các module chúng ta đã viết
from config import Config
from dataset import DSB2018Dataset
from transform import get_transforms
from modelUnet import UNet
from utils import seed_everything, dice_coeff

def train_model():
    # 1. Cài đặt cơ bản
    cfg = Config()
    seed_everything(cfg.SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Device: {device}")

    # 2. Load Data
    full_dataset = DSB2018Dataset(
        root_dir=cfg.TRAIN_DIR,
        transform=get_transforms(cfg, stage='train') # Augmentation
    )

    # Chia Train/Val (90% Train, 10% Val)
    n_val = int(len(full_dataset) * 0.1)
    n_train = len(full_dataset) - n_val
    train_set, val_set = random_split(full_dataset, [n_train, n_val])
    
    # Lưu ý: Val set nên tắt Augmentation (chỉ resize), nhưng ở đây ta chia random
    # từ full_dataset đã gắn transform. Để đơn giản ta chấp nhận aug ở val.
    # Trong thực tế kỹ hơn, ta sẽ tạo 2 dataset riêng biệt.

    train_loader = DataLoader(train_set, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=True)
    
    print(f"📦 Data: {n_train} Train images | {n_val} Val images")

    # 3. Khởi tạo Model
    # n_channels=3 (RGB), n_classes=1 (Binary: Cell vs Background)
    model = UNet(n_channels=3, n_classes=1, bilinear=True).to(device)

    # 4. Optimizer & Loss
    optimizer = optim.RMSprop(model.parameters(), lr=1e-4, weight_decay=1e-8, momentum=0.9)
    # Hoặc dùng Adam: optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Scheduler: Giảm LR nếu loss không giảm
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2) 
    
    # Loss: Kết hợp BCEWithLogitsLoss (tốt hơn BCELoss thuần)
    criterion = nn.BCEWithLogitsLoss()

    # 5. Training Loop
    EPOCHS = 50 # Số vòng lặp train (tăng lên 50-100 nếu cần)
    best_dice = 0.0

    print("\n🏁 START TRAINING...")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        
        # --- TRAIN ---
        with tqdm(total=len(train_set), desc=f'Epoch {epoch+1}/{EPOCHS}', unit='img') as pbar:
            for batch in train_loader:
                images, masks = batch
                images = images.to(device, dtype=torch.float32)
                masks = masks.to(device, dtype=torch.float32)

                # Forward
                masks_pred = model(images)
                loss = criterion(masks_pred, masks)

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Log
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                pbar.update(images.shape[0])

        # --- VALIDATION ---
        model.eval()
        val_dice = 0
        with torch.no_grad():
            for batch in val_loader:
                images, masks = batch
                images = images.to(device, dtype=torch.float32)
                masks = masks.to(device, dtype=torch.float32)

                masks_pred = model(images)
                
                # Chuyển logits -> xác suất (sigmoid) -> nhị phân (threshold 0.5)
                pred_prob = torch.sigmoid(masks_pred)
                pred_binary = (pred_prob > 0.5).float()
                
                # Tính Dice
                val_dice += dice_coeff(pred_binary, masks)

        val_score = val_dice / len(val_loader)
        
        # Update Scheduler (theo dõi Dice Score)
        scheduler.step(val_score)

        print(f"📊 Epoch {epoch+1} Result: Loss = {epoch_loss/len(train_loader):.4f} | Val Dice = {val_score:.4f}")

        # Save Best Model
        if val_score > best_dice:
            best_dice = val_score
            torch.save(model.state_dict(), 'best_unet_checkpoint.pth')
            print(f"💾 Saved Best Model (Dice: {best_dice:.4f})")

    print("\n✅ TRAINING COMPLETED.")

if __name__ == '__main__':
    train_model()