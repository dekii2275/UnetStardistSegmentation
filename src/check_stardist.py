from dataset_stardist import StarDistDataset
from config import Config
from transform import get_transforms
import matplotlib.pyplot as plt
import torch

cfg = Config()
ds = StarDistDataset(cfg.TRAIN_DIR, transform=get_transforms(cfg, stage='train'), n_rays=32)

# Lấy thử 1 mẫu
img, prob, dist = ds[0]

print("Image shape:", img.shape)       # (3, 256, 256)
print("Prob shape:", prob.shape)       # (1, 256, 256)
print("Dist shape:", dist.shape)       # (32, 256, 256) 

# Vẽ thử
plt.figure(figsize=(12, 4))
plt.subplot(131); plt.imshow(img.permute(1,2,0)); plt.title("Input Image")
plt.subplot(132); plt.imshow(prob.squeeze(), cmap='gray'); plt.title("Prob Map")
plt.subplot(133); plt.imshow(dist[0], cmap='jet'); plt.title("Dist Map (Ray 0)") # Xem tia góc 0 độ
plt.show()