import official
import os
import json
import numpy as np
import csv
import cv2



from tqdm import tqdm

# ================= CẤU HÌNH ĐƯỜNG DẪN =================
VIDEO_DIR = "archive/videos"                  # Thư mục chứa ~12k video .mp4
OUTPUT_FEATURE_DIR = "features_npy/"   # Nơi lưu vector .npy của từng video
CSV_OUTPUT = "wlasl_2000_dataset.csv"  # File CSV đầu ra chứa nhãn và đường dẫn

CLASS_LIST_FILE = "archive/wlasl_class_list.txt"
SPLIT_FULL_FILE = "archive/nslt_2000.json"
WLASL_META_FILE = "archive/WLASL_v0.3.json"

MAX_FRAMES = 60  # Chuẩn hóa số lượng frame cho mỗi video (T=60)

os.makedirs(OUTPUT_FEATURE_DIR, exist_ok=True)


def extract_features_from_video(video_path: str, start_frame: int = 1, end_frame: int = -1) -> np.ndarray:
    """
    Đọc video, cắt theo khoảng frame của cử chỉ, trích xuất vector 141 chiều
    trên từng frame và chuẩn hóa độ dài về MAX_FRAMES x 141.
    
    :param video_path: Đường dẫn tới file video .mp4
    :param start_frame: Frame bắt đầu của ký hiệu (1-indexed)
    :param end_frame: Frame kết thúc của ký hiệu (-1 nếu lấy tới cuối)
    :return: Numpy array shape (MAX_FRAMES, 141) hoặc None nếu video lỗi/không đọc được
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    sequence = []
    frame_idx = 1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Chỉ xử lý các frame nằm trong khoảng cử chỉ được gán nhãn
        if start_frame <= frame_idx and (end_frame == -1 or frame_idx <= end_frame):
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = official.holistic.process(rgb_frame)
            
            # Trích xuất vector 141 phần tử từ hàm có sẵn của bạn
            feat = official.extract_holistic_features(results)
            sequence.append(feat)

        frame_idx += 1
        if end_frame != -1 and frame_idx > end_frame:
            break

    cap.release()

    # Trường hợp video rỗng hoặc không trích xuất được frame nào
    if len(sequence) == 0:
        return None

    sequence = np.array(sequence, dtype=np.float32)  # Shape: (N_frames, 141)

    # Chuẩn hóa số lượng frame về MAX_FRAMES (T=60)
    total_frames = len(sequence)
    if total_frames >= MAX_FRAMES:
        # Lấy mẫu đều 60 frame dọc theo độ dài video
        indices = np.linspace(0, total_frames - 1, MAX_FRAMES, dtype=int)
        normalized_seq = sequence[indices]
    else:
        # Zero-padding nếu video ngắn hơn 60 frame
        pad_len = MAX_FRAMES - total_frames
        normalized_seq = np.pad(sequence, ((0, pad_len), (0, 0)), mode='constant', constant_values=0.0)

    return normalized_seq  # Output shape: (60, 141)


# ================= 1. NẠP METADATA & MAPPING =================

# Map: class_id (int) -> gloss (str)
id_to_gloss = {}
with open(CLASS_LIST_FILE, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) == 2:
            id_to_gloss[int(parts[0])] = parts[1]

# Map: video_id -> {subset, class_id, start_frame, end_frame} từ nslt_2000.json
with open(SPLIT_FULL_FILE, "r", encoding="utf-8") as f:
    nslt_data = json.load(f)

# ================= 2. VÒNG LẶP TRÍCH XUẤT & GÁN NHÃN =================

csv_records = []
all_videos = [f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")]

print(f"Bắt đầu xử lý {len(all_videos)} video...")

for f_name in tqdm(all_videos, desc="Extracting & Labeling"):
    video_id = os.path.splitext(f_name)[0]
    
    # Kiểm tra xem video có nằm trong tập 2000 nhãn không
    if video_id not in nslt_data:
        continue
    
    meta = nslt_data[video_id]
    class_id = int(meta["action"][0])          # Nhãn 0 -> 1999
    start_frame = int(meta["action"][1])       # Frame bắt đầu ký
    end_frame = int(meta["action"][2])         # Frame kết thúc ký
    subset = meta["subset"]                    # train / val / test
    gloss = id_to_gloss.get(class_id, "unknown")

    video_path = os.path.join(VIDEO_DIR, f_name)
    feature_save_path = os.path.join(OUTPUT_FEATURE_DIR, f"{video_id}.npy")

    # Trích xuất và lưu vector nếu chưa tồn tại
    if not os.path.exists(feature_save_path):
        try:
            vector_data = extract_features_from_video(video_path, start_frame, end_frame)
            if vector_data is not None and len(vector_data) > 0:
                np.save(feature_save_path, vector_data)
            else:
                continue
        except Exception as e:
            print(f"Lỗi khi xử lý video {video_id}: {e}")
            continue

    # Thêm bản ghi vào bảng dữ liệu
    csv_records.append({
        "video_id": video_id,
        "class_id": class_id,
        "gloss": gloss,
        "subset": subset,
        "feature_path": feature_save_path
    })

# ================= 3. XUẤT FILE CSV =================

fieldnames = ["video_id", "class_id", "gloss", "subset", "feature_path"]

with open(CSV_OUTPUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(csv_records)