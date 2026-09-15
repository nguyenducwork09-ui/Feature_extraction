import copy
import torch
from model import SignLanguageMSTCN

# --- 1. LẤY TRỌNG SỐ TRUNG BÌNH QUA 5 FOLDS (WEIGHT AVERAGING) ---
# Load fold 1 và deepcopy để tạo tensor độc lập
first_dict = torch.load("best_mstcn_fold_1.pt", map_location="cpu")
avg_state_dict = copy.deepcopy(first_dict)

# Chuyển các tensor float sang float32 để tích lũy
for key in avg_state_dict:
    if avg_state_dict[key].is_floating_point():
        avg_state_dict[key] = avg_state_dict[key].clone().to(torch.float32)

# Cộng dồn trọng số từ các fold còn lại
for fold in range(2, 6):
    state_dict = torch.load(f"best_mstcn_fold_{fold}.pt", map_location="cpu")
    for key in avg_state_dict:
        if avg_state_dict[key].is_floating_point():
            avg_state_dict[key] += state_dict[key]

# Chia trung bình cho 5
for key in avg_state_dict:
    if avg_state_dict[key].is_floating_point():
        avg_state_dict[key] /= 5.0

# Lưu checkpoint hợp nhất
torch.save(avg_state_dict, "final_mstcn_averaged.pt")
print("Đã tạo model trung bình thành công: final_mstcn_averaged.pt")


# --- 2. KHỞI TẠO MÔ HÌNH VÀ EXPORT SANG ONNX ---
# Tự động lấy số class từ tầng classifier mà không cần hardcode
num_classes = avg_state_dict["classifier.bias"].shape[0]
feature_dim = 188
hidden_dim = 128

print(f"Khởi tạo mô hình với {num_classes} classes, feature_dim={feature_dim}...")
model = SignLanguageMSTCN(in_features=feature_dim, hidden_dim=hidden_dim, num_classes=num_classes)
model.load_state_dict(avg_state_dict)
model.eval()

# Dummy input: [Batch=1, Time=60 frames, Channels=188]
dummy_input = torch.randn(1, 60, feature_dim, dtype=torch.float32)

onnx_output_path = "sign_language_mstcn.onnx"

torch.onnx.export(
    model,
    dummy_input,
    onnx_output_path,
    export_params=True,
    opset_version=18,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size", 1: "num_frames"},   # Cho phép số frame T thay đổi tự do theo video
        "output": {0: "batch_size"}
    }
)
print(f"Đã xuất mô hình ONNX thành công: {onnx_output_path}")