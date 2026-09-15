import cv2
import numpy as np
import os
import onnxruntime as ort
ort.set_default_logger_severity(3)
from rtmlib import Wholebody, draw_skeleton

# 1. Khởi tạo RTMPose Wholebody chạy trực tiếp trên GPU CUDA (RTX 3050)
# mode='balanced' cân bằng tốt giữa tốc độ và độ chính xác
# Muốn siêu nhanh trên edge thì đổi thành mode='lightweight'


# Bắt buộc trỏ về CUDA v12.8 để khớp với DLL cu12 của onnxruntime-gpu
cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"
cudnn_bin = r"C:\Users\maxvn\AppData\Local\Programs\Python\Python314\Lib\site-packages\nvidia\cudnn\bin"
cublas_bin = r"C:\Users\maxvn\AppData\Local\Programs\Python\Python314\Lib\site-packages\nvidia\cublas\bin"

for p in [cuda_bin, cudnn_bin, cublas_bin]:
    if os.path.exists(p):
        os.environ["PATH"] = p + os.pathsep + os.environ["PATH"]
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(p)
            except Exception:
                pass




def extract_rtmpose_features(keypoints, scores, thr=0.8):
    # Nếu không phát hiện được người nào
    if len(keypoints) == 0:
        return np.zeros(94)

    # Lọc ra người ngồi gần nhất (khung xương có diện tích lớn nhất)
    max_area = 0
    target_idx = 0

    for i in range(len(keypoints)):
        # Chỉ tính toán trên các điểm nhận diện rõ (score > thr)
        valid_pts = keypoints[i][scores[i] > thr]
        
        if len(valid_pts) > 0:
            min_vals = np.min(valid_pts, axis=0)
            max_vals = np.max(valid_pts, axis=0)
            
            # Tính diện tích (width * height)
            area = (max_vals[0] - min_vals[0]) * (max_vals[1] - min_vals[1])
            
            if area > max_area:
                max_area = area
                target_idx = i

    # Gán kpts và scs cho người gần nhất
    kpts = keypoints[target_idx]
    scs = scores[target_idx]

    # --- PHẦN 1: POSE VỊ TRÍ SO VỚI TRUNG ĐIỂM VAI ---
    # RTMPose index: Nose(0), L_Shoulder(5), R_Shoulder(6), L_Elbow(7), R_Elbow(8), L_Wrist(9), R_Wrist(10)
    key_indices = [0, 5, 6, 7, 8, 9, 10]
    pose_features = np.zeros(len(key_indices) * 2)

    # Kiểm tra độ tin cậy của 2 vai
    if scs[5] > thr and scs[6] > thr:
        mid_shoulder = (kpts[5] + kpts[6]) / 2.0
        shoulder_dist = np.linalg.norm(kpts[5] - kpts[6])
        if shoulder_dist < 1e-6:
            shoulder_dist = 1.0

        extracted_pose = []
        for idx in key_indices:
            if scs[idx] > thr:
                norm_pt = (kpts[idx] - mid_shoulder) / shoulder_dist
            else:
                norm_pt = np.array([0.0, 0.0])
            extracted_pose.extend(norm_pt)
        pose_features = np.array(extracted_pose)

    # --- PHẦN 2 & 3: HÌNH DẠNG 2 BÀN TAY ---
    def get_hand_shape(start_idx, end_idx, base_idx, mcp_idx):
        # start_idx đến end_idx là 21 điểm của bàn tay
        hand_pts = kpts[start_idx:end_idx]
        hand_scs = scs[start_idx:end_idx]

        # Cổ tay không detect được -> tay bị khuất
        if scs[base_idx] < thr:
            return np.zeros(40)

        base_pt = kpts[base_idx]
        palm_size = np.linalg.norm(kpts[mcp_idx] - base_pt)
        if palm_size < 1e-6:
            palm_size = 1.0

        features = []
        # Lấy từ điểm 1 đến 20 (bỏ cổ tay vì luôn là gốc 0, 0)
        for i in range(start_idx + 1, end_idx):
            if scs[i] > thr:
                norm_pt = (kpts[i] - base_pt) / palm_size
            else:
                norm_pt = np.array([0.0, 0.0])
            features.extend(norm_pt)
        return np.array(features)

    # Tay trái: index 91-111, cổ tay: 91, gốc ngón giữa: 95
    left_hand = get_hand_shape(91, 112, 91, 95)
    # Tay phải: index 112-132, cổ tay: 112, gốc ngón giữa: 116
    right_hand = get_hand_shape(112, 133, 112, 116)

    return np.concatenate([pose_features, left_hand, right_hand])


def is_valid_features(features: np.ndarray, tol: float = 1e-6) -> bool:
    return not np.allclose(features, 0.0, atol=tol)


def save_vector_to_csv(vector, file_path="dataset.csv", label="A"):
    vector_str = ','.join(map(str, vector))
    with open(file_path, 'a') as f:
        f.write(f"{vector_str},{label}\n")


cap = cv2.VideoCapture(0)

if __name__ == "__main__":
    wholebody = Wholebody(mode='lightweight', device='cuda')
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # Trích xuất 133 điểm với RTX 3050
        keypoints, scores = wholebody(frame)

        # Trích xuất vector chuẩn hóa 94 phần tử
        vector_X = extract_rtmpose_features(keypoints, scores)

        if is_valid_features(vector_X):
            save_vector_to_csv(vector_X, file_path="dataset.csv", label="A")

        # Vẽ khung xương 133 điểm lên ảnh
        img_show = draw_skeleton(frame, keypoints, scores, kpt_thr=0.4)

        cv2.imshow("RTMPose Wholebody Tracking (CUDA)", img_show)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()