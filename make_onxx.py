import urllib.request
import zipfile
import os

url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
task_file = "hand_landmarker.task"

print("Đang tải model gốc từ Google...")
urllib.request.urlretrieve(url, task_file)

# Trích xuất file .tflite chuẩn
with zipfile.ZipFile(task_file, "r") as z:
    for name in z.namelist():
        if "landmarks_detector" in name or name.endswith(".tflite"):
            z.extract(name, ".")
            # Đổi tên thành hand_landmark.tflite cho gọn
            if os.path.exists("hand_landmark.tflite"):
                os.remove("hand_landmark.tflite")
            os.rename(name, "hand_landmark.tflite")
            print(f"Đã trích xuất thành công: hand_landmark.tflite ({os.path.getsize('hand_landmark.tflite') / (1024*1024):.2f} MB)")
            break

if os.path.exists(task_file):
    os.remove(task_file)