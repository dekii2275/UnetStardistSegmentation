import os

class Config:
    # --- ĐƯỜNG DẪN DỮ LIỆU ---
    # Bạn thay đổi đường dẫn này trỏ đúng đến folder dataset của bạn
    DATA_DIR = './data' 
    TRAIN_DIR = os.path.join(DATA_DIR, 'train')
    TEST_DIR = os.path.join(DATA_DIR, 'test')
    
    # --- THAM SỐ MÔ HÌNH & TRAIN ---
    SEED = 42
    IMG_SIZE = 256        # Kích thước đầu vào mô hình
    BATCH_SIZE = 8       # Tùy VRAM GPU, 16 hoặc 32 là ổn
    NUM_WORKERS = 4       # Số luồng load dữ liệu
    
    # --- CHIẾN LƯỢC XỬ LÝ DỮ LIỆU ---
    # Các tùy chọn: 'resize', 'random', 'smart_roi', 'multiscale_smart'
    CROP_STRATEGY = 'multiscale_smart' 
    ENABLE_AUG = True     # Bật/Tắt augmentation (Xoay, lật, nhiễu...)