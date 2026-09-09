import official
import os
import json
import numpy as np
import csv
import cv2
from concurrent.futures import ProcessPoolExecutor, as_completed
import onnxruntime as ort
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

os.environ["ORT_LOGGING_LEVEL"] = "3"
ort.set_default_logger_severity(3)
NUM_WORKERS = 8


from tqdm import tqdm

# ================= CẤU HÌNH ĐƯỜNG DẪN =================
VIDEO_DIR = "archive/videos"                  # Thư mục chứa ~12k video .mp4
OUTPUT_FEATURE_DIR = "features_npy/"   # Nơi lưu vector .npy của từng video
CSV_OUTPUT = "wlasl_2000_dataset.csv"  # File CSV đầu ra chứa nhãn và đường dẫn

CLASS_LIST_FILE = "archive/wlasl_class_list.txt"
SPLIT_FULL_FILE = "archive/nslt_2000.json"
WLASL_META_FILE = "archive/WLASL_v0.3.json"

# Khai báo biến worker_wholebody ở mức module (mỗi process con sẽ tự giữ 1 bản riêng)
worker_wholebody = None

MAX_FRAMES = 60  # Chuẩn hóa số lượng frame cho mỗi video (T=60)

os.makedirs(OUTPUT_FEATURE_DIR, exist_ok=True)


# multiproc
def init_worker():
    """Hàm khởi tạo model độc lập cho từng tiến trình con"""
    global worker_wholebody
    import os
    import onnxruntime as ort
    ort.set_default_logger_severity(3)
    import official
    from rtmlib import Wholebody
    os.environ["ORT_LOGGING_LEVEL"] = "3"

    cudnn_dir = r"C:\Users\maxvn\AppData\Local\Programs\Python\Python314\Lib\site-packages\nvidia\cudnn\bin"
    cuda_dir = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"

    if os.path.exists(cudnn_dir):
        os.add_dll_directory(cudnn_dir)
    if os.path.exists(cuda_dir):
        os.add_dll_directory(cuda_dir)

    worker_wholebody = Wholebody(
        to_openpose=False,
        mode='lightweight',
        backend='onnxruntime',
        device='cuda'
    )

def extract_features_from_video(video_path: str, start_frame: int = 1, end_frame: int = -1) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Xác định chính xác phạm vi ký
    s_frame = max(1, start_frame)
    e_frame = end_frame if (end_frame != -1 and end_frame <= total_video_frames) else total_video_frames
    
    if e_frame < s_frame:
        e_frame = total_video_frames

    total_valid_frames = e_frame - s_frame + 1
    if total_valid_frames <= 0:
        cap.release()
        return None

    # 1. Tính toán trước đúng các index frame cần lấy
    if total_valid_frames >= MAX_FRAMES:
        target_indices = set(np.linspace(s_frame, e_frame, MAX_FRAMES, dtype=int))
    else:
        target_indices = set(range(s_frame, e_frame + 1))

    # Nhảy tới frame đầu tiên cần đọc
    cap.set(cv2.CAP_PROP_POS_FRAMES, s_frame - 1)
    
    current_frame_idx = s_frame
    sequence = []

    while cap.isOpened() and current_frame_idx <= e_frame:
        ret, frame = cap.read()
        if not ret:
            break

        # CHỈ CHẠY GPU TRÊN ĐÚNG CÁC FRAME NẰM TRONG TARGET_INDICES
        if current_frame_idx in target_indices:
            # Resize nhẹ để giảm nghẽn bus PCIe nếu video độ phân giải cao
            h, w = frame.shape[:2]
            if w > 640:
                scale = 640.0 / w
                frame = cv2.resize(frame, (640, int(h * scale)))

            keypoints, scores = worker_wholebody(frame)
            feat = official.extract_rtmpose_features(keypoints, scores)
            sequence.append(feat)

        current_frame_idx += 1

    cap.release()

    if len(sequence) == 0:
        return None

    sequence = np.array(sequence, dtype=np.float32)

    # Pad nếu video ngắn hơn MAX_FRAMES
    if len(sequence) < MAX_FRAMES:
        pad_len = MAX_FRAMES - len(sequence)
        sequence = np.pad(sequence, ((0, pad_len), (0, 0)), mode='edge')

    return sequence[:MAX_FRAMES]

def process_single_task(task_data):
    """Worker nhận 1 task, gọi trích xuất và trả về thông tin ghi CSV"""
    video_id, f_name, start_frame, end_frame, class_id, subset, gloss = task_data
    video_path = os.path.join(VIDEO_DIR, f_name)
    feature_save_path = os.path.join(OUTPUT_FEATURE_DIR, f"{video_id}.npy")

    # Bỏ qua nếu video này đã trích xuất từ trước (tránh chạy lại khi ngắt quãng)
    if os.path.exists(feature_save_path):
        return {
            "video_id": video_id,
            "class_id": class_id,
            "gloss": gloss,
            "subset": subset,
            "feature_path": feature_save_path
        }

    try:
        features = extract_features_from_video(video_path, start_frame, end_frame)
        if features is not None and len(features) > 0:
            np.save(feature_save_path, features)
            return {
                "video_id": video_id,
                "class_id": class_id,
                "gloss": gloss,
                "subset": subset,
                "feature_path": feature_save_path
            }
    except Exception as e:
        print(f"\n[LỖI WORKER] {video_id}: {e}")
        return None

    return None


if __name__ == "__main__":
    # 1. NẠP METADATA & CLASS MAPPING
    id_to_gloss = {}
    if os.path.exists(CLASS_LIST_FILE):
        with open(CLASS_LIST_FILE, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    id_to_gloss[int(parts[0])] = parts[1]

    with open(SPLIT_FULL_FILE, "r", encoding="utf-8") as f:
        nslt_data = json.load(f)

    # 2. LẬP DANH SÁCH CÁC VIDEO CẦN XỬ LÝ
    all_videos = [f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")]
    task_list = []

    for f_name in all_videos:
        video_id = os.path.splitext(f_name)[0]
        if video_id in nslt_data:
            meta = nslt_data[video_id]
            class_id = int(meta["action"][0])
            start_frame = int(meta["action"][1])
            end_frame = int(meta["action"][2])
            subset = meta["subset"]
            gloss = id_to_gloss.get(class_id, "unknown")

            task_list.append((video_id, f_name, start_frame, end_frame, class_id, subset, gloss))

    print(f"Tổng số video hợp lệ trong dataset: {len(task_list)}")
    print(f"Bắt đầu khởi chạy song song với {NUM_WORKERS} workers...")

    # 3. KÍCH HOẠT MULTIPROCESSING
    csv_records = []
    with ProcessPoolExecutor(max_workers=NUM_WORKERS, initializer=init_worker) as executor:
        futures = {executor.submit(process_single_task, task): task for task in task_list}
        
        for future in tqdm(as_completed(futures), total=len(task_list), desc="Extracting Features"):
            res = future.result()
            if res is not None:
                csv_records.append(res)

    with ProcessPoolExecutor(max_workers=NUM_WORKERS, initializer=init_worker) as executor:
        # Dùng as_completed để video nào xong là cập nhật thanh tiến trình ngay
        futures = {executor.submit(process_single_task, task): task for task in task_list}
        
        for future in tqdm(as_completed(futures), total=len(task_list), desc="Extracting Features"):
            res = future.result()
            if res is not None:
                csv_records.append(res)

    fieldnames = ["video_id", "class_id", "gloss", "subset", "feature_path"]
    with open(CSV_OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_records)

    print(f"\nĐã hoàn thành! Đã lưu {len(csv_records)} vector vào {OUTPUT_FEATURE_DIR}")