import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import adaptive_normalization

class DSB2018Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Đường dẫn đến folder dataset (VD: stage1_train)
            transform (callable, optional): Augmentation pipeline.
        """
        self.root_dir = root_dir
        self.transform = transform
        # Lấy danh sách ID các ảnh
        self.ids = next(os.walk(root_dir))[1]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id_ = self.ids[idx]
        
        # 1. Đường dẫn file
        img_path = os.path.join(self.root_dir, id_, 'images', id_ + '.png')
        mask_dir = os.path.join(self.root_dir, id_, 'masks')
        
        # 2. Load Ảnh
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 3. Load Mask (Gộp nhiều file mask con thành 1 mask lớn)
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if os.path.exists(mask_dir):
            for f in next(os.walk(mask_dir))[2]:
                m = cv2.imread(os.path.join(mask_dir, f), cv2.IMREAD_GRAYSCALE)
                if m is not None:
                    # Resize mask nếu cần (đề phòng dữ liệu lỗi size)
                    if m.shape != (h, w):
                        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                    mask = np.maximum(mask, m)
        
        # Chuyển mask về nhị phân (0 và 1) nếu chưa phải
        # Albumentations thích mask kiểu float cho binary segmentation đôi khi, 
        # nhưng chuẩn là giữ nguyên giá trị rồi xử lý sau. 
        # Ở đây ta giữ mask nền đen (0) và vật thể (255) hoặc (1).
        # Để an toàn cho tính toán loss sau này, ta đưa về 0.0 và 1.0
        mask = (mask > 0).astype(np.float32)

        # 4. Áp dụng Adaptive Normalization (Quan trọng!)
        image = adaptive_normalization(image)

        # 5. Áp dụng Augmentation (Transform)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        # Mask đầu ra của ToTensorV2 thường không có channel dimension ở cuối nếu là 2D
        # Ta cần unsqueeze để thành (1, H, W) cho đúng chuẩn Pytorch Segmentation
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
            
        return image, mask