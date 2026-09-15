import torch
import torch.nn as nn

class DilatedResidualBlock1D(nn.Module):
    """
    Khối tích chập 1D giãn nở với đường nối tắt (Residual Connection).
    Mỗi tầng mở rộng trường tiếp nhận (receptive field) qua hệ số dilation d.
    """
    def __init__(self, channels, dilation):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation
        )
        self.bn = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        res = x
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        return out + res


class MSTCNStage(nn.Module):
    """
    Một giai đoạn (Single Stage) tích hợp chuỗi các khối Dilated Convolutions
    với dilation tăng dần theo cấp số nhân: 1, 2, 4, 8, 16.
    """
    def __init__(self, in_features, hidden_dim, out_features, num_layers=5):
        super().__init__()
        self.input_conv = nn.Conv1d(in_features, hidden_dim, kernel_size=1)
        
        self.layers = nn.ModuleList([
            DilatedResidualBlock1D(channels=hidden_dim, dilation=2**i)
            for i in range(num_layers)
        ])
        
        self.output_conv = nn.Conv1d(hidden_dim, out_features, kernel_size=1)

    def forward(self, x):
        # x: [Batch, In_Features, Time]
        out = self.input_conv(x)
        for layer in self.layers:
            out = layer(out)
        out = self.output_conv(out)
        return out


class SignLanguageMSTCN(nn.Module):
    """
    Mô hình nhận diện hoàn chỉnh cho vector đặc trưng 94 chiều.
    - Stage 1: Dự đoán phân bố xác suất ban đầu qua thời gian.
    - Stage 2: Tinh chỉnh (refinement) để khử nhiễu giữa các frame liên tiếp.
    - Global Pooling: Phân loại cử chỉ cho cả chuỗi video.
    """
    def __init__(self, in_features=94, hidden_dim=64, num_classes=100, num_layers_per_stage=4):
        super().__init__()
        self.num_classes = num_classes

        # Giai đoạn 1: Dự đoán chuỗi đặc trưng thô
        self.stage1 = MSTCNStage(
            in_features=in_features, 
            hidden_dim=hidden_dim, 
            out_features=hidden_dim, 
            num_layers=num_layers_per_stage
        )
        
        # Giai đoạn 2: Tinh chỉnh đặc trưng đa tỷ lệ
        self.stage2 = MSTCNStage(
            in_features=hidden_dim, 
            hidden_dim=hidden_dim, 
            out_features=hidden_dim, 
            num_layers=num_layers_per_stage
        )

        # Global Temporal Pooling & Classifier
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # Input shape từ pipeline: [Batch, Time_steps, 94]
        # Chuyển đổi sang chuẩn Conv1d: [Batch, Channels=94, Time_steps]
        x = x.transpose(1, 2)

        out_s1 = self.stage1(x)
        out_s2 = self.stage2(out_s1)

        # Gom toàn bộ đặc trưng thời gian về 1 vector biểu diễn
        pooled = self.pool(out_s2).squeeze(-1) # [Batch, hidden_dim]
        logits = self.classifier(pooled)       # [Batch, num_classes]

        return logits

if __name__ == "__main__":
    # Giả lập batch 8 video, mỗi video được trích xuất 60 frame, mỗi frame 94 số
    batch_size = 8
    time_steps = 60
    feature_dim = 94
    num_classes = 50  # Số từ vựng cử chỉ cần nhận diện

    model = SignLanguageMSTCN(in_features=feature_dim, hidden_dim=256, num_classes=num_classes)
    sample_input = torch.randn(batch_size, time_steps, feature_dim)

    output = model(sample_input)
    print("Input Shape :", sample_input.shape)  # torch.Size([8, 60, 94])
    print("Output Shape:", output.shape)        # torch.Size([8, 50])