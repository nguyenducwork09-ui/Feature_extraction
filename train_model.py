import os
import copy
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import SignLanguageMSTCN


class SignLanguageDataset(Dataset):
    def __init__(self, file_paths, labels, compute_velocity=True):
        self.file_paths = file_paths
        self.labels = labels
        self.compute_velocity = compute_velocity

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        data = np.load(path)  # Shape: (T, 94)

        # Trích xuất thêm kênh vận tốc (Velocity: dx = x_t - x_{t-1})
        if self.compute_velocity:
            if len(data) > 1:
                vel = np.diff(data, axis=0, prepend=data[0:1])
            else:
                vel = np.zeros_like(data)
            data = np.concatenate([data, vel], axis=-1)  # Tăng lên (T, 188)

        x = torch.tensor(data, dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


def pad_collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = [len(seq) for seq in sequences]
    max_len = max(lengths)
    feature_dim = sequences[0].shape[-1]

    padded_seqs = torch.zeros(len(sequences), max_len, feature_dim, dtype=torch.float32)
    for i, seq in enumerate(sequences):
        padded_seqs[i, :len(seq), :] = seq

    labels = torch.stack(labels)
    return padded_seqs, labels


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x_batch.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)

        total_loss += loss.item() * x_batch.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

    return total_loss / total, correct / total


if __name__ == "__main__":
    # --- CẤU HÌNH 8 LUỒNG TÍNH TOÁN CỦA PYTORCH ---
    NUM_WORKERS = 8
    torch.set_num_threads(8)  # Ép CPU tính toán ma trận/tensor trên 8 core song song
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csv_path = "wlasl_2000_dataset.csv"

    # --- 1. NẠP DỮ LIỆU ---
    df = pd.read_csv(csv_path)
    all_file_paths = df["feature_path"].values

    label_encoder = LabelEncoder()
    all_labels = label_encoder.fit_transform(df["class_id"].values)
    num_classes = len(label_encoder.classes_)
    
    print(f"Tổng mẫu: {len(all_file_paths)} | Classes: {num_classes} | Thiết bị: {device}")
    print(f"Khởi chạy pipeline với {NUM_WORKERS} workers đa luồng...\n")

    # --- 2. THIẾT LẬP HUẤN LUYỆN ---
    K_FOLDS = 5
    NUM_EPOCHS = 60
    BATCH_SIZE = 64        # Tăng lên 64 để tận dụng luồng nạp dữ liệu nhanh hơn
    LEARNING_RATE = 1e-3
    FEATURE_DIM = 188      # 94 tọa độ + 94 vận tốc

    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(all_file_paths, all_labels)):
        print(f"========== FOLD {fold + 1}/{K_FOLDS} ==========")

        train_set = SignLanguageDataset(all_file_paths[train_idx], all_labels[train_idx], compute_velocity=True)
        val_set = SignLanguageDataset(all_file_paths[val_idx], all_labels[val_idx], compute_velocity=True)

        # 8 worker song song + persistent_workers để không bị overhead tạo/hủy tiến trình mỗi epoch
        train_loader = DataLoader(
            train_set, 
            batch_size=BATCH_SIZE, 
            shuffle=True, 
            collate_fn=pad_collate_fn, 
            num_workers=NUM_WORKERS, 
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )
        val_loader = DataLoader(
            val_set, 
            batch_size=BATCH_SIZE, 
            shuffle=False, 
            collate_fn=pad_collate_fn, 
            num_workers=NUM_WORKERS, 
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )

        model = SignLanguageMSTCN(in_features=FEATURE_DIM, hidden_dim=128, num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

        best_val_acc = 0.0
        best_weights = None

        for epoch in range(NUM_EPOCHS):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            scheduler.step(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_weights = copy.deepcopy(model.state_dict())

            print(f"Epoch [{epoch+1:02d}/{NUM_EPOCHS}] "
                  f"Train Loss: {train_loss:.4f} - Acc: {train_acc*100:.2f}% | "
                  f"Val Loss: {val_loss:.4f} - Acc: {val_acc*100:.2f}%")

        print(f">> Fold {fold + 1} Best Val Acc: {best_val_acc * 100:.2f}%\n")
        torch.save(best_weights, f"best_mstcn_fold_{fold+1}.pt")
        fold_results.append(best_val_acc)

    print("================ TỔNG KẾT CROSS VALIDATION ================")
    print(f"Mean Acc qua {K_FOLDS} folds: {np.mean(fold_results)*100:.2f}% ± {np.std(fold_results)*100:.2f}%")