import cv2
import mediapipe as mp
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_drawing
import numpy as np
import time

img = cv2.imread("./imgTest.jpg")

# Mediapipe hands model

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)



#====COORDINATE EXTRACTION FUNCTION====
def get_landmark_coordinates(landmarks):
    """
    Trích xuất tọa độ cố định 120 phần tử:
    - 60 phần tử đầu: Tay Trái (Left)
    - 60 phần tử sau: Tay Phải (Right)
    Nếu thiếu tay nào, tự động điền toàn bộ giá trị 0 cho tay đó.
    """
    left_hand = np.zeros(60)
    right_hand = np.zeros(60)

    if results.multi_hand_landmarks and results.multi_handedness:
        # Duyệt song song landmarks và nhãn Left/Right
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Nhãn trả về là 'Left' hoặc 'Right'
            label = handedness.classification[0].label

            # Tọa độ gốc cổ tay (ID 0)
            base_x = hand_landmarks.landmark[0].x
            base_y = hand_landmarks.landmark[0].y
            base_z = hand_landmarks.landmark[0].z

            features = []
            # Bỏ qua điểm 0, lấy từ điểm 1 đến 20 (20 * 3 = 60 giá trị)
            for lm in hand_landmarks.landmark[1:]:
                features.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])

            # Gán vào mảng tương ứng
            if label == 'Left':
                left_hand = np.array(features)
            elif label == 'Right':
                right_hand = np.array(features)

    # Ghép 2 tay lại thành 1 vector duy nhất có shape (120,)
    return np.concatenate([left_hand, right_hand])




last_scan = 0
current_vector = None

def process_periodic_scan(results, last_scan_time, interval=2.0):
    """
    Hàm kiểm tra và quét tọa độ sau mỗi chu kỳ (mặc định 2 giây).
    
    :param results: Đối tượng trả về từ hands.process()
    :param last_scan_time: Mốc time.time() của lần quét trước
    :param interval: Khoảng thời gian giữa các lần quét (giây)
    :return: (is_scanned, vector_coords, updated_scan_time, time_left)
             - is_scanned (bool): True nếu vừa quét thành công
             - vector_coords (np.array hoặc None): Vector X (63 phần tử) nếu quét được
             - updated_scan_time (float): Cập nhật lại mốc thời gian
             - time_left (float): Số giây còn lại cho lần quét tiếp theo
    """
    now = time.time()
    elapsed = now - last_scan_time
    time_left = max(0.0, interval - elapsed)

    # Chưa đủ thời gian chu kỳ
    if elapsed < interval:
        return False, None, last_scan_time, time_left

    # Đã đủ thời gian, kiểm tra xem có bàn tay trong khung hình không
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        vector = get_landmark_coordinates(hand_landmarks.landmark)
        return True, vector, now, 0.0

    # Đã đủ 2 giây nhưng không thấy bàn tay
    return False, None, now, 0.0


def save_vector_to_csv(vector, file_path="dataset.csv", label="A"):
    """
    Lưu vector 120 phần tử vào file text/csv, có kèm nhãn ở cột cuối cùng.
    """
    # Chuyển vector thành chuỗi các số ngăn cách bởi dấu phẩy
    vector_str = ",".join([f"{val:.6f}" for val in vector])
    
    # Ghi thêm vào cuối file (append)
    with open(file_path, mode="a", encoding="utf-8") as f:
        # Định dạng: val1,val2,val3,...,val120,label
        f.write(f"{vector_str},{label}\n")


def normalize_landmarks(vector_120):
    """
    Chuẩn hóa kích thước (Scale Invariance) cho vector 120 phần tử.
    - 60 phần tử đầu: Tay Trái
    - 60 phần tử sau: Tay Phải
    
    Thước đo chuẩn: khoảng cách từ Cổ tay (ID 0) đến Gốc ngón giữa (ID 9).
    Vì đã dời gốc về cổ tay (0,0,0), ID 9 nằm tại index 24, 25, 26 trong mảng 60 số.
    """
    vector_norm = np.copy(vector_120)

    # Tách riêng 2 tay: mỗi tay 60 phần tử
    left_hand = vector_norm[:60]
    right_hand = vector_norm[60:]

    # Hàm phụ chuẩn hóa cho 1 bàn tay (60 phần tử)
    def scale_single_hand(hand_coords):
        # Nếu tay vắng mặt (toàn số 0) thì bỏ qua
        if not np.any(hand_coords != 0):
            return hand_coords

        # Vì bỏ qua điểm 0 (cổ tay), các điểm từ 1 -> 20 sẽ có index = (ID - 1) * 3
        # Điểm ID 9 (gốc ngón giữa) nằm tại: (9 - 1) * 3 = index 24, 25, 26
        x9, y9, z9 = hand_coords[24], hand_coords[25], hand_coords[26]

        # Khoảng cách từ cổ tay (0, 0, 0) đến ID 9
        palm_size = np.sqrt(x9**2 + y9**2 + z9**2)

        # Chia tỉ lệ nếu khoảng cách hợp lệ (tránh chia cho 0)
        if palm_size > 1e-6:
            hand_coords = hand_coords / palm_size

        return hand_coords

    # Chuẩn hóa độc lập cho từng tay
    left_norm = scale_single_hand(left_hand)
    right_norm = scale_single_hand(right_hand)

    # Ghép lại thành vector 120 phần tử hoàn chỉnh
    return np.concatenate([left_norm, right_norm])

#==============================DEBUG DRAWING FUNCTION========================================
def draw_hand_landmarks_debug(frame, hand_landmarks, show_coords=False):
    """
    Vẽ các điểm mốc, số thứ tự ID và tọa độ lên khung hình.
    :param frame: Khung hình OpenCV (BGR)
    :param hand_landmarks: Đối tượng landmarks của một bàn tay từ MediaPipe
    :param show_coords: True nếu muốn in chi tiết tọa độ (x, y) lên ảnh
    """
    h, w, _ = frame.shape

    for lm_id, lm in enumerate(hand_landmarks.landmark):
        # 1. Chuyển đổi tọa độ chuẩn hóa sang pixel
        cx, cy = int(lm.x * w), int(lm.y * h)

        # 2. Phân loại màu sắc để dễ quan sát:
        # ID 0 (Cổ tay): Đỏ
        # ID 4, 8, 12, 16, 20 (Đầu ngón tay): Vàng
        # Các khớp còn lại: Xanh lá
        if lm_id == 0:
            point_color = (0, 0, 255)       # Đỏ
        elif lm_id in [4, 8, 12, 16, 20]:
            point_color = (0, 255, 255)     # Vàng
        else:
            point_color = (0, 255, 0)       # Xanh lá

        # Vẽ chấm tròn tại điểm khớp
        cv2.circle(frame, (cx, cy), 6, point_color, cv2.FILLED)

        # 3. Hiển thị thông tin chữ
        if show_coords:
            # Hiển thị cả ID và tọa độ pixel (cx, cy)
            label = f"{lm_id}:({cx},{cy})"
            cv2.putText(frame, label, (cx + 8, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        else:
            # Chỉ hiển thị số ID cho gọn, tránh rối mắt
            cv2.putText(frame, str(lm_id), (cx + 8, cy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    return frame





#================================MAIN LOOP========================================
if __name__ == "__main__":
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and find hands
        results = hands.process(rgb_frame)

        # Draw hand landmarks on the frame
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS
            )

                draw_hand_landmarks_debug(frame, hand_landmarks, show_coords=False)


        # Save to file dataset
        vector_coords = normalize_landmarks(get_landmark_coordinates(results.multi_hand_landmarks[0].landmark) if results.multi_hand_landmarks else np.zeros(120))
        if vector_coords is not None:
            save_vector_to_csv(vector_coords, file_path="dataset.csv", label="A")
        # Display the resulting frame
        cv2.imshow('Hand Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()