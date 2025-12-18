import os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from scipy.ndimage import maximum_filter, label
from skimage.draw import polygon
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Import modules từ project của bạn ---
from config import Config
from dataset_stardist import StarDistDataset
from transform import get_transforms
from model_stardist import StarDist
from utils import seed_everything

# ==========================
# 1. CÁC HÀM HẬU XỬ LÝ (POST-PROCESSING)
# ==========================

def find_peaks(prob_map, prob_thresh=0.5, min_distance=2):
    """Tìm local maxima trên bản đồ xác suất."""
    H, W = prob_map.shape
    mask = prob_map >= prob_thresh
    if not np.any(mask):
        return []

    size = 2 * min_distance + 1
    footprint = np.ones((size, size), dtype=bool)
    max_filt = maximum_filter(prob_map, footprint=footprint, mode="nearest")
    peaks = (prob_map == max_filt) & mask

    ys, xs = np.nonzero(peaks)
    coords = list(zip(ys.tolist(), xs.tolist()))
    return coords

def rays_to_polygon_mask(center_y, center_x, rays, H, W):
    """Chuyển đổi tia (rays) thành mask đa giác."""
    n_rays = rays.shape[0]
    max_radius = max(H, W)
    rays = np.clip(rays, 0.0, float(max_radius))

    angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    ys = center_y + rays * np.sin(angles)
    xs = center_x + rays * np.cos(angles)

    ys = np.clip(ys, 0, H - 1)
    xs = np.clip(xs, 0, W - 1)

    rr, cc = polygon(ys, xs, (H, W))
    mask = np.zeros((H, W), dtype=bool)
    mask[rr, cc] = True
    return mask

def nms_on_masks(masks, scores, iou_thresh=0.3, min_size=0):
    """Non-Maximum Suppression trên danh sách các mask."""
    if len(masks) == 0: return []

    areas = np.array([m.sum() for m in masks], dtype=np.float32)
    # Lọc mask quá nhỏ
    keep_initial = [i for i, a in enumerate(areas) if a >= min_size]
    if len(keep_initial) == 0: return []

    masks = [masks[i] for i in keep_initial]
    scores = np.asarray(scores, dtype=np.float32)[keep_initial]
    areas = areas[keep_initial]

    order = np.argsort(scores)[::-1]
    keep = []
    used = np.zeros(len(masks), dtype=bool)

    for idx in order:
        if used[idx]: continue
        keep.append(idx)
        used[idx] = True
        m_i = masks[idx]
        area_i = areas[idx]

        for j in order:
            if used[j]: continue
            m_j = masks[j]
            inter = np.logical_and(m_i, m_j).sum()
            if inter == 0: continue
            union = area_i + areas[j] - inter
            iou = inter / union
            if iou > iou_thresh:
                used[j] = True

    kept_global = [keep_initial[i] for i in keep]
    return kept_global

def stardist_postprocess(prob_map, dist_map, prob_thresh=0.5, peak_min_distance=2, nms_iou_thresh=0.3):
    """
    Quy trình hậu xử lý đầy đủ:
    Prob Map -> Peaks -> Polygons -> NMS -> Instance Mask
    """
    H, W = prob_map.shape
    centers = find_peaks(prob_map, prob_thresh=prob_thresh, min_distance=peak_min_distance)
    
    if len(centers) == 0:
        return np.zeros((H, W), dtype=np.int32), {}

    masks = []
    scores = []

    # Dist map của bạn có shape (n_rays, H, W) -> cần transpose khi truy cập từng điểm
    # Nhưng trong loop bên dưới ta truy cập dist_map[:, y, x] là đúng với shape (n_rays, H, W)
    
    for (y, x) in centers:
        rays = dist_map[:, y, x] # (n_rays,)
        if np.all(rays <= 0): continue
        
        mask = rays_to_polygon_mask(y, x, rays, H, W)
        if mask.sum() == 0: continue
        
        masks.append(mask)
        scores.append(prob_map[y, x])

    keep_idxs = nms_on_masks(masks, scores, iou_thresh=nms_iou_thresh)

    labeled = np.zeros((H, W), dtype=np.int32)
    inst_scores = {}
    next_id = 1
    
    # Vẽ đè các mask lên nhau (cái nào điểm cao hơn được vẽ trước, nhưng ở đây vẽ đè theo thứ tự keep)
    # Để tối ưu hiển thị, ta nên vẽ ngược lại hoặc chỉ cần đảm bảo ID khác nhau
    for ki in keep_idxs:
        labeled[masks[ki]] = next_id
        inst_scores[next_id] = float(scores[ki])
        next_id += 1

    return labeled, inst_scores

# ==========================
# 2. CÁC HÀM TÍNH TOÁN METRIC (AP, IoU)
# ==========================

def collect_ap_data_for_image(gt_mask, pred_mask, inst_scores, iou_thresh=0.5):
    gt_ids = [i for i in np.unique(gt_mask) if i != 0]
    pred_ids = [i for i in np.unique(pred_mask) if i != 0]

    num_gt = len(gt_ids)
    if num_gt == 0 and len(pred_ids) == 0: return [], [], 0
    if num_gt == 0: return [inst_scores.get(i, 0) for i in pred_ids], [0]*len(pred_ids), 0

    # Sort pred theo score giảm dần
    pred_ids_sorted = sorted(pred_ids, key=lambda pid: inst_scores.get(pid, 0.0), reverse=True)

    gt_used = set()
    scores = []
    tp_flags = []

    for pid in pred_ids_sorted:
        p_mask = (pred_mask == pid)
        best_iou = 0.0
        best_gid = None

        for gid in gt_ids:
            if gid in gt_used: continue
            g_mask = (gt_mask == gid)
            inter = np.logical_and(p_mask, g_mask).sum()
            if inter == 0: continue
            union = p_mask.sum() + g_mask.sum() - inter
            iou = inter / union
            if iou > best_iou:
                best_iou = iou
                best_gid = gid

        scores.append(inst_scores.get(pid, 0.0))
        if best_iou >= iou_thresh and best_gid is not None:
            tp_flags.append(1)
            gt_used.add(best_gid)
        else:
            tp_flags.append(0)

    return scores, tp_flags, num_gt

def compute_ap_from_scores(all_scores, all_tp_flags, total_gt):
    if total_gt == 0: return 0.0
    if len(all_scores) == 0: return 0.0

    scores = np.asarray(all_scores, dtype=np.float32)
    tp_flags = np.asarray(all_tp_flags, dtype=np.int32)

    order = np.argsort(-scores)
    tp = tp_flags[order]
    fp = 1 - tp

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    recalls = tp_cum / float(total_gt)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap)

# ==========================
# 3. HÀM MAIN
# ==========================

def evaluate():
    # 1. Setup
    cfg = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Device: {device} | Batch Size: 1 (Eval mode)")
    seed_everything(cfg.SEED)

    # 2. Load Model
    # n_rays=32 (Cần khớp với lúc train)
    model = StarDist(n_channels=3, n_rays=32).to(device)
    
    checkpoint_path = 'best_stardist_checkpoint.pth'
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"✅ Loaded checkpoint: {checkpoint_path}")
    except FileNotFoundError:
        print(f"⚠️ Checkpoint not found at {checkpoint_path}. Using random init.")

    model.eval()

    # 3. Load Data (Validation Split)
    # Lưu ý: Dataset của bạn trả về: image, target_prob, target_dist
    full_dataset = StarDistDataset(
        root_dir=cfg.TRAIN_DIR,
        transform=get_transforms(cfg, stage='test'),
        n_rays=32
    )
    
    n_val = int(len(full_dataset) * 0.1)
    n_train = len(full_dataset) - n_val
    _, val_set = random_split(full_dataset, [n_train, n_val])
    
    # Batch size 1 để dễ xử lý từng ảnh và tránh OOM khi post-process
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
    print(f"📦 Evaluating on {len(val_set)} images...")

    # 4. Evaluation Loop
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    all_scores = {t: [] for t in iou_thresholds}
    all_tp_flags = {t: [] for t in iou_thresholds}
    total_gt_for_ap = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Eval"):
            # Lấy data (batch size = 1)
            images, prob_gt_tensor, dist_gt_tensor = batch
            
            images = images.to(device)
            
            # Predict
            pred_logits, pred_dists = model(images)
            pred_probs = torch.sigmoid(pred_logits)
            
            # Chuyển sang Numpy
            # image (B, C, H, W), prob (B, 1, H, W), dist (B, Rays, H, W)
            # Lấy phần tử đầu tiên vì batch=1
            prob_pred_np = pred_probs[0, 0].cpu().numpy()
            dist_pred_np = pred_dists[0].cpu().numpy() # (Rays, H, W)
            
            prob_gt_np = prob_gt_tensor[0, 0].cpu().numpy()
            
            # --- TẠO GROUND TRUTH INSTANCE MASK ---
            # Dataset hiện tại của bạn không trả về instance mask gốc (label 1, 2, 3...)
            # Nó chỉ trả về prob (0/1) và dist.
            # Ta dùng thuật toán Connected Components để tái tạo instance mask từ prob_gt
            binary_gt = prob_gt_np > 0.5
            gt_mask, num_gt = label(binary_gt) # scipy.ndimage.label
            
            # --- POST-PROCESS PREDICTION ---
            # Chuyển (prob, dist) -> (instance mask, scores)
            pred_mask, inst_scores = stardist_postprocess(
                prob_pred_np, 
                dist_pred_np, 
                prob_thresh=0.5,     # Ngưỡng xác suất
                peak_min_distance=2, # Khoảng cách tối thiểu giữa các đỉnh
                nms_iou_thresh=0.3   # Ngưỡng đè nhau NMS
            )
            
            # --- TÍNH TOÁN AP ---
            # Update tổng số lượng ground truth
            # num_gt có thể lấy từ hàm label() ở trên hoặc max của gt_mask
            gt_count = len(np.unique(gt_mask)) - 1 # Trừ background 0
            total_gt_for_ap += gt_count

            for t in iou_thresholds:
                scores_img, tp_flags_img, _ = collect_ap_data_for_image(
                    gt_mask, pred_mask, inst_scores, iou_thresh=t
                )
                all_scores[t].extend(scores_img)
                all_tp_flags[t].extend(tp_flags_img)

    # 5. Tính kết quả cuối cùng
    ap_values = []
    print("\n📊 === RESULTS ===")
    for t in iou_thresholds:
        ap = compute_ap_from_scores(all_scores[t], all_tp_flags[t], total_gt_for_ap)
        ap_values.append(ap)
        print(f"  AP @ IoU {t:.2f}: {ap:.4f}")

    mAP = np.mean(ap_values) if len(ap_values) > 0 else 0.0
    print(f"\n🏆 mAP @ [0.5:0.95]: {mAP:.4f}")
    
    # 6. Vẽ biểu đồ
    plt.figure(figsize=(10, 6))
    plt.plot(iou_thresholds, ap_values, 'o-', color='purple', linewidth=2)
    plt.axhline(y=mAP, color='red', linestyle='--', label=f'Mean AP = {mAP:.4f}')
    plt.title('StarDist mAP Evaluation', fontsize=14)
    plt.xlabel('IoU Threshold')
    plt.ylabel('Average Precision')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    vis_path = 'evaluation_stardist_result.png'
    plt.savefig(vis_path)
    print(f"📈 Chart saved to: {vis_path}")

if __name__ == "__main__":
    evaluate()