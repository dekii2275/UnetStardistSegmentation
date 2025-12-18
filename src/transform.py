import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms(cfg, stage='train'):
    """
    Trả về pipeline augmentation.
    stage: 'train' (có augmentation) hoặc 'valid'/'test' (chỉ resize/crop chuẩn)
    """
    transforms_list = []
    
    # --- GIAI ĐOẠN 1: SPATIAL SAMPLING (CẮT/RESIZE) ---
    if stage == 'train':
        if cfg.CROP_STRATEGY == 'resize':
            transforms_list.append(A.Resize(cfg.IMG_SIZE, cfg.IMG_SIZE))
            
        elif cfg.CROP_STRATEGY == 'random':
            transforms_list.extend([
                A.PadIfNeeded(min_height=cfg.IMG_SIZE, min_width=cfg.IMG_SIZE, border_mode=0, value=0),
                A.RandomCrop(height=cfg.IMG_SIZE, width=cfg.IMG_SIZE, p=1.0)
            ])
            
        elif cfg.CROP_STRATEGY == 'smart_roi':
            transforms_list.extend([
                A.PadIfNeeded(min_height=cfg.IMG_SIZE, min_width=cfg.IMG_SIZE, border_mode=0, value=0),
                A.OneOf([
                    A.CropNonEmptyMaskIfExists(height=cfg.IMG_SIZE, width=cfg.IMG_SIZE, p=0.8),
                    A.RandomCrop(height=cfg.IMG_SIZE, width=cfg.IMG_SIZE, p=0.2),
                ], p=1.0)
            ])
            
        elif cfg.CROP_STRATEGY == 'multiscale_smart':
            transforms_list.extend([
                # Random Scale nhẹ (+/- 10%) trước khi cắt
                A.RandomScale(scale_limit=0.1, p=0.5), 
                A.PadIfNeeded(min_height=cfg.IMG_SIZE, min_width=cfg.IMG_SIZE, border_mode=0, value=0),
                A.OneOf([
                    A.CropNonEmptyMaskIfExists(height=cfg.IMG_SIZE, width=cfg.IMG_SIZE, p=0.8),
                    A.RandomCrop(height=cfg.IMG_SIZE, width=cfg.IMG_SIZE, p=0.2),
                ], p=1.0)
            ])
    else:
        # Với Validation/Test thì chỉ Resize về kích thước chuẩn
        transforms_list.append(A.Resize(cfg.IMG_SIZE, cfg.IMG_SIZE))

    # --- GIAI ĐOẠN 2: MORPHOLOGICAL AUGMENTATION (Chỉ train) ---
    if stage == 'train' and cfg.ENABLE_AUG:
        transforms_list.extend([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            
            # Biến dạng nhẹ
            A.ElasticTransform(alpha=1, sigma=20, alpha_affine=20, p=0.2),
            
            # Nhiễu nhẹ (như hạt bụi)
            A.CoarseDropout(max_holes=3, max_height=8, max_width=8, p=0.2),
            
            # Chỉnh sáng tối nhẹ
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.2),
            
            # Nhiễu hạt mịn
            A.GaussNoise(var_limit=(10.0, 30.0), p=0.1),
        ])

    # --- GIAI ĐOẠN 3: CHUYỂN VỀ TENSOR ---
    # Lưu ý: Normalize về [0, 1] cho float32
    transforms_list.append(A.ToFloat(max_value=255.0)) 
    transforms_list.append(ToTensorV2())
    
    return A.Compose(transforms_list)