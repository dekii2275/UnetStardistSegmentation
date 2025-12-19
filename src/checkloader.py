import torch
from torch.utils.data import DataLoader
from config import Config
from dataset import DSB2018Dataset
from transform import get_transforms
from utils import seed_everything, visualize_batch

if __name__ == '__main__':
    # 1. Setup
    cfg = Config()
    seed_everything(cfg.SEED)
    print(f"✅ Config loaded. Strategy: {cfg.CROP_STRATEGY}")

    try:
        train_dataset = DSB2018Dataset(
            root_dir=cfg.TRAIN_DIR,
            transform=get_transforms(cfg, stage='train')
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=cfg.BATCH_SIZE, 
            shuffle=True, 
            num_workers=cfg.NUM_WORKERS
        )
        
        print(f"Dataset created. Total samples: {len(train_dataset)}")
        
        # 3. Lấy thử 1 batch
        images, masks = next(iter(train_loader))
        
        print(f"📦 Batch shape info:")
        print(f"   - Images: {images.shape}  (Batch, Channel, Height, Width)")
        print(f"   - Masks:  {masks.shape}")
        print(f"   - Image Value Range: [{images.min():.2f}, {images.max():.2f}]")
        print(f"   - Mask Unique Values: {torch.unique(masks)}")
        
        # 4. Visualize
        print("🖼️ Displaying batch preview...")
        visualize_batch((images, masks), title=f"Strategy: {cfg.CROP_STRATEGY}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Gợi ý: Kiểm tra lại đường dẫn TRAIN_DIR trong file config.py")