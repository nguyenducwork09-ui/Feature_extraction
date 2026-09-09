import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from mediapipe.tasks.python.vision import holistic_landmarker as mp_holistic
from mediapipe.tasks.python.vision import drawing_utils as mp_drawing
import time



# PP
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index Finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle Finger
    (9, 10), (10, 11), (11, 12),
    # Ring Finger
    (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm Knuckle Connections
    (5, 9), (9, 13), (13, 17)
]


CONFIDENCE_VALUE = 0.8

# 1. Khởi tạo Holistic
base_options = python.BaseOptions(model_asset_path='holistic_landmarker.task')
options = vision.HolisticLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    output_segmentation_mask=True,
    running_mode=vision.RunningMode.VIDEO,
    min_pose_detection_confidence = CONFIDENCE_VALUE,
    min_face_detection_confidence = CONFIDENCE_VALUE,
    min_hand_landmarks_confidence = CONFIDENCE_VALUE,
)

holistic = vision.HolisticLandmarker.create_from_options(options)

def extract_holistic_features(results):
    """
    Trích xuất vector toàn diện:
    - Phần 1: Tọa độ các điểm mốc Pose quan trọng so với trung điểm vai (21 giá trị)
    - Phần 2: Hình dạng tay trái đã chuẩn hóa (60 giá trị)
    - Phần 3: Hình dạng tay phải đã chuẩn hóa (60 giá trị)
    Tổng cộng vector: 21 + 60 + 60 = 141 phần tử
    """
    
    # --- PHẦN 1: POSE (Vị trí tay so với cơ thể) ---
    pose_features = np.zeros(7 * 3) # 7 điểm: Mũi(0), Vai T(11), Vai P(12), Cùi chỏ T(13), Cùi chỏ P(14), Cổ tay T(15), Cổ tay P(16)
    
    if results.pose_landmarks:
        pose = results.pose_landmarks
        
        # Gốc quy chiếu: Trung điểm 2 vai
        mid_shoulder_x = (pose[11].x + pose[12].x) / 2.0
        mid_shoulder_y = (pose[11].y + pose[12].y) / 2.0
        mid_shoulder_z = (pose[11].z + pose[12].z) / 2.0
        
        # Thước đo chuẩn: Độ rộng vai (tránh người ngồi gần/xa)
        shoulder_dist = np.sqrt((pose[11].x - pose[12].x)**2 + 
                                (pose[11].y - pose[12].y)**2 + 
                                (pose[11].z - pose[12].z)**2)
        if shoulder_dist < 1e-6: shoulder_dist = 1.0

        key_indices = [0, 11, 12, 13, 14, 15, 16] # 7 điểm mốc quan trọng phần thân trên
        extracted_pose = []
        for idx in key_indices:
            extracted_pose.extend([
                (pose[idx].x - mid_shoulder_x) / shoulder_dist,
                (pose[idx].y - mid_shoulder_y) / shoulder_dist,
                (pose[idx].z - mid_shoulder_z) / shoulder_dist
            ])
        pose_features = np.array(extracted_pose)

    # --- PHẦN 2: HÌNH DẠNG CHI TIẾT 2 BÀN TAY ---
    def get_hand_shape(hand_landmarks):
        if not hand_landmarks:
            return np.zeros(60)
        
        base_x, base_y, base_z = hand_landmarks[0].x, hand_landmarks[0].y, hand_landmarks[0].z
        # Đoạn chuẩn: Cổ tay (0) đến khớp gốc ngón giữa (9)
        mcp9 = hand_landmarks[9]
        palm_size = np.sqrt((mcp9.x - base_x)**2 + (mcp9.y - base_y)**2 + (mcp9.z - base_z)**2)
        if palm_size < 1e-6: palm_size = 1.0

        features = []
        for lm in hand_landmarks[1:]:
            features.extend([
                (lm.x - base_x) / palm_size,
                (lm.y - base_y) / palm_size,
                (lm.z - base_z) / palm_size
            ])
        return np.array(features)

    left_hand = get_hand_shape(results.left_hand_landmarks)
    right_hand = get_hand_shape(results.right_hand_landmarks)

    # Ghép toàn bộ thành 1 vector duy nhất 141 phần tử
    return np.concatenate([pose_features, left_hand, right_hand])


def is_valid_features(features: np.ndarray, tol: float = 1e-6) -> bool:
    """
    Kiểm tra vector đặc trưng.
    Trả về:
        - False: nếu TẤT CẢ các giá trị đều là 0.0 (hoặc xấp xỉ 0 do sai số float)
        - True: nếu có ít nhất một giá trị khác 0 (có dữ liệu hợp lệ)
    """
    # np.allclose kiểm tra xem toàn bộ mảng có bằng 0 hay không
    if np.allclose(features, 0.0, atol=tol):
        return False
    return True



def save_vector_to_csv(vector, file_path="dataset.csv", label="A"):
    """
    Lưu vector vào file CSV với nhãn tương ứng.
    
    :param vector: Vector dữ liệu (1D numpy array)
    :param file_path: Đường dẫn file CSV
    :param label: Nhãn ký hiệu (ví dụ: 'A', 'B', ...)
    """
    vector_str = ','.join(map(str, vector))
    with open(file_path, 'a') as f:
        f.write(f"{vector_str},{label}\n")

cap = cv2.VideoCapture(0)



# Official to extract

if __name__ == "__main__":
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        # Xử lý toàn bộ cơ thể + 2 tay cùng lúc
        timestamp_ms = int(time.time() * 1000)
        results = holistic.detect_for_video(mp_image, timestamp_ms=timestamp_ms)
        
        # 1. Vẽ Pose
        mp_drawing.draw_landmarks(frame, results.pose_landmarks)
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks)
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks)

        # Lấy vector X tổng hợp (141 phần tử)
        vector_X = extract_holistic_features(results)

        if is_valid_features(vector_X):
            save_vector_to_csv(vector_X, file_path="dataset.csv", label="A")

        cv2.imshow("Sign Language Holistic Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    holistic.close()