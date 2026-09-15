import cv2
import numpy as np
import onnxruntime as ort
from collections import deque
from rtmlib import Wholebody, draw_skeleton


# Import hàm trích xuất của bạn
from official import extract_rtmpose_features

class RealTimeSignLanguagePredictor:
    def __init__(self, onnx_path, label_encoder_path=None):
        # Chạy ONNX MS-TCN trên CPU cho nhẹ, nhường toàn bộ GPU cho RTMPose
        self.session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.class_names = np.load(label_encoder_path, allow_pickle=True) if label_encoder_path else None

    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    def predict(self, feature_buffer):
        data = np.array(feature_buffer)  # (T, 94)
        vel = np.diff(data, axis=0, prepend=data[0:1]) if len(data) > 1 else np.zeros_like(data)
        
        final_features = np.concatenate([data, vel], axis=-1)
        input_tensor = final_features[np.newaxis, ...].astype(np.float32)
        
        logits = self.session.run(None, {self.input_name: input_tensor})[0]
        probs = self.softmax(logits)[0]
        
        predicted_idx = np.argmax(probs)
        label = self.class_names[predicted_idx] if self.class_names is not None else f"Class_{predicted_idx}"
        return label, probs[predicted_idx] * 100


# =====================================================================
# CHƯƠNG TRÌNH CHÍNH
# =====================================================================
if __name__ == "__main__":
    # Khởi tạo mô hình RTMPose
    # (Tôi thêm dòng này vì trong code của bạn bị thiếu biến wholebody)
    wholebody = Wholebody(mode='balanced', backend='onnxruntime', device='cuda')
    
    # Khởi tạo mô hình nhận diện ngôn ngữ ký hiệu
    predictor = RealTimeSignLanguagePredictor("sign_language_mstcn.onnx")

    # Cấu hình cửa sổ trượt
    WINDOW_SIZE = 60
    PREDICT_FREQ = 2
    CONF_THRESHOLD = 50.0

    buffer = deque(maxlen=WINDOW_SIZE)
    current_label = "Waiting..."
    current_conf = 0.0
    frame_count = 0

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_count += 1

        # 1. Trích xuất khung xương bằng rtmlib (Dữ liệu thật)
        keypoints, scores = wholebody(frame)

        # 2. Trích xuất đặc trưng 94 chiều từ khung xương
        vector_X = extract_rtmpose_features(keypoints, scores)
        buffer.append(vector_X)

        # 3. Đưa vào ONNX dự đoán nếu đủ 60 frames
        if len(buffer) == WINDOW_SIZE and frame_count % PREDICT_FREQ == 0:
            label, conf = predictor.predict(buffer)
            # Tạm thời cập nhật kết quả liên tục bất chấp độ tự tin cao hay thấp
            current_label = label
            current_conf = conf

        # 4. Vẽ khung xương 133 điểm lên ảnh
        img_show = draw_skeleton(frame, keypoints, scores, kpt_thr=0.4)

        # 5. Hiển thị UI kết quả lên trên cùng
        cv2.rectangle(img_show, (10, 10), (600, 100), (0, 0, 0), -1)
        
        color = (0, 255, 0) if current_conf > 70 else (0, 165, 255)
        cv2.putText(img_show, f"{current_label} ({current_conf:.1f}%)", (20, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(img_show, f"Buffer: {len(buffer)}/{WINDOW_SIZE}", (20, 85), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("RTMPose + Sign Language AI", img_show)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()