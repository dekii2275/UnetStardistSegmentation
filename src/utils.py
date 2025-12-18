import random
import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def adaptive_normalization(img):
    """
    Chuẩn hóa ảnh theo phân vị (Percentile Normalization)
    Giúp tăng tương phản cho ảnh y tế tối/mờ.
    """
    img = img.astype(np.float32)
    
    # Lấy phân vị 1% (nền tối) và 99.8% (điểm sáng nhất không phải nhiễu)
    low = np.percentile(img, 1)
    high = np.percentile(img, 99.8)
    
    # Công thức chuẩn hóa: (I - P_low) / (P_high - P_low)
    img_norm = (img - low) / (high - low + 1e-7)
    
    # Kẹp giá trị trong khoảng [0, 1] và chuyển về [0, 255] uint8
    img_norm = np.clip(img_norm, 0, 1)
    return (img_norm * 255).astype(np.uint8)

def visualize_batch(batch, title="Batch Preview"):
    """Hàm vẽ nhanh một batch để kiểm tra"""
    images, masks = batch
    batch_size = len(images)
    
    plt.figure(figsize=(16, 8))
    for i in range(min(batch_size, 4)): # Vẽ tối đa 4 ảnh
        plt.subplot(2, 4, i+1)
        img = images[i].permute(1, 2, 0).numpy()
        # Denormalize để hiển thị nếu cần, ở đây giả sử ảnh đã là 0-1
        plt.imshow(img)
        plt.title(f"Image {i}")
        plt.axis('off')
        
        plt.subplot(2, 4, i+5)
        mask = masks[i].squeeze().numpy()
        plt.imshow(mask, cmap='gray')
        plt.title(f"Mask {i}")
        plt.axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    output_filename = "check_batch_preview.png"
    plt.savefig(output_filename)
    print(f"🖼️ Đã lưu ảnh kiểm tra tại: {os.path.abspath(output_filename)}")
    plt.close()

def dice_coeff(pred, target, smooth=1e-5):
    """Tính chỉ số Dice (F1-Score) cho Binary Segmentation"""
    # pred: output của sigmoid (0-1)
    # target: mask (0 hoặc 1)
    
    # Làm phẳng ảnh thành vector 1 chiều
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice.item()