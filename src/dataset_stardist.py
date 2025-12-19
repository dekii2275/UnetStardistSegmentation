import os
import cv2
import numpy as np
import torch
import stardist
from torch.utils.data import Dataset
from stardist.geometry import star_dist 
from utils import adaptive_normalization

class StarDistDataset(Dataset):
    def __init__(self, root_dir, transform=None, n_rays=32):
        """
        Args:
            root_dir (string): Đường dẫn data.
            transform (callable): Augmentation pipeline.
            n_rays (int): Số lượng tia (mặc định 32 cho StarDist).
        """
        self.root_dir = root_dir
        self.transform = transform
        self.n_rays = n_rays
        
        # Kiểm tra path
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"❌ Không tìm thấy: {root_dir}")
            
        self.ids = next(os.walk(root_dir))[1]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id_ = self.ids[idx]
        
        # 1. Đường dẫn
        img_path = os.path.join(self.root_dir, id_, 'images', id_ + '.png')
        mask_dir = os.path.join(self.root_dir, id_, 'masks')
        
        # 2. Load Ảnh & Chuẩn hóa (Adaptive Normalization)
        image = cv2.imread(img_path)
        if image is None:
             raise FileNotFoundError(f"Lỗi ảnh: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # --- ÁP DỤNG TIỀN XỬ LÝ SÁNG ---
        image = adaptive_normalization(image)

        # 3. Load Mask (QUAN TRỌNG: Cần Instance Label Mask)
        # Với StarDist, các tế bào dính nhau phải có ID khác nhau (1, 2, 3...)
        # chứ không phải gộp hết thành 1 như U-Net.
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.int32) # Dùng int32 để lưu ID
        
        if os.path.exists(mask_dir):
            # Load từng file mask con và gán ID riêng biệt
            for i, f in enumerate(next(os.walk(mask_dir))[2], start=1):
                m = cv2.imread(os.path.join(mask_dir, f), cv2.IMREAD_GRAYSCALE)
                if m is not None:
                    if m.shape != (h, w):
                        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                    # Gán label i vào vị trí có tế bào
                    mask[m > 0] = i
        
        # 4. Augmentation (Albumentations)
        # Lưu ý: Transform của bạn đang trả về Tensor, ta cần xử lý khéo léo
        if self.transform:
            # Albumentations cần mask là numpy
            # image lúc này là uint8 [0-255], mask là int32 [0-N]
            augmented = self.transform(image=image, mask=mask)
            
            image_tensor = augmented['image'] # Tensor (3, H, W)
            mask_tensor = augmented['mask']   # Tensor (H, W) hoặc (1, H, W)
            
            # Chuyển mask augmented về lại Numpy int32 để tính StarDist
            # (Vì hàm star_dist chỉ chạy trên CPU numpy)
            if torch.is_tensor(mask_tensor):
                mask_np = mask_tensor.squeeze().cpu().numpy().astype(np.int32)
            else:
                mask_np = mask_tensor.astype(np.int32)
        else:
            # Nếu không aug thì convert thủ công
            import albumentations as A
            from albumentations.pytorch import ToTensorV2
            basic_trans = A.Compose([A.ToFloat(max_value=255.0), ToTensorV2()])
            res = basic_trans(image=image)
            image_tensor = res['image']
            mask_np = mask

        # 5. TÍNH TOÁN STAR-DIST TARGETS
        # Đầu vào: Mask chứa các ID (0, 1, 2...)
        # Đầu ra: 
        #   - distances: (H, W, n_rays) -> Khoảng cách
        #   - object_probs: (H, W) -> Xác suất tâm
        
        try:
            # Tính khoảng cách hình sao (Nặng nhất ở bước này)
            distances = star_dist(mask_np, self.n_rays, mode='cpp') 
        except Exception:
            # Fallback nếu chưa cài stardist cpp extension
            distances = star_dist(mask_np, self.n_rays, mode='python')

        # Tạo mask xác suất (Probabilities map) - Tương tự binary mask
        object_probs = (mask_np > 0).astype(np.float32)

        # 6. Chuyển đổi định dạng cho PyTorch
        # Distances từ (H, W, n_rays) -> (n_rays, H, W)
        distances = np.transpose(distances, (2, 0, 1)).astype(np.float32)
        distances_tensor = torch.from_numpy(distances)
        
        object_probs_tensor = torch.from_numpy(object_probs).unsqueeze(0) # (1, H, W)

        # Trả về 3 thứ: Ảnh, Prob Map (Mask), Distance Map
        return image_tensor, object_probs_tensor, distances_tensor