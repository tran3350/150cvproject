"""
Step 4: Train a GRU classifier on pose-feature windows.

Architecture — PoseGRU:
  Input  : (B, T=64, F=175)
  GRU    : 2 layers, hidden=128
  Dropout: 0.3
  Linear : hidden → n_classes (4)

Training details:
  - Class-weighted cross-entropy (handles "none" dominance)
  - AdamW lr=3e-4, weight_decay=1e-3
  - CosineAnnealingLR over 30 epochs
  - Best val-accuracy checkpoint saved

Saves: models/judo_pose_gru.pt  (includes input_dim for inference)
"""
import json
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Loaded from meta.json at runtime — do not hardcode here
LABELS: list[str] = []


# ---------- dataset ----------------------------------------------------------

class PoseWindows(Dataset):
    def __init__(self, x_path, y_path):
        self.X = np.load(x_path).astype(np.float32)   # (N, T, F)
        self.y = np.load(y_path).astype(np.int64)      # (N,)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.tensor(self.y[i])


# ---------- model ------------------------------------------------------------

class PoseGRU(nn.Module):
    def __init__(self, input_dim, hidden=128, n_layers=2, n_classes=4, dropout=0.3):
        super().__init__()
        self.gru  = nn.GRU(
            input_dim, hidden, n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x):
        # x : (B, T, F)
        _, h = self.gru(x)            # h : (n_layers, B, hidden)
        return self.head(self.drop(h[-1]))   # (B, n_classes)


# ---------- eval -------------------------------------------------------------

def eval_model(model, loader):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(DEVICE))
            ps.append(logits.argmax(1).cpu().numpy())
            ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


# ---------- main -------------------------------------------------------------

def main():
    global LABELS
    with open("data/pose_windows/meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    LABELS = meta["labels"]

    tr_ds  = PoseWindows("data/pose_windows/X_train.npy", "data/pose_windows/y_train.npy")
    val_ds = PoseWindows("data/pose_windows/X_val.npy",   "data/pose_windows/y_val.npy")

    tr_loader  = DataLoader(tr_ds,  batch_size=32, shuffle=True,  num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    # Class weights — inverse frequency
    y_tr    = np.load("data/pose_windows/y_train.npy")
    counts  = np.bincount(y_tr, minlength=len(LABELS)).astype(np.float32)
    weights = torch.tensor(counts.sum() / np.maximum(counts, 1.0), device=DEVICE)

    input_dim = tr_ds.X.shape[-1]          # 175
    model     = PoseGRU(input_dim).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optim     = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=30)

    print(f"Device: {DEVICE}  |  input_dim={input_dim}  |  train={len(tr_ds)}  val={len(val_ds)}")

    os.makedirs("models", exist_ok=True)
    best_acc = 0.0

    for epoch in range(1, 31):
        model.train()
        total_loss = 0.0
        for x, yb in tr_loader:
            x, yb = x.to(DEVICE), yb.to(DEVICE)
            optim.zero_grad()
            loss = criterion(model(x), yb)
            loss.backward()
            optim.step()
            total_loss += loss.item() * x.size(0)

        scheduler.step()

        y_true, y_pred = eval_model(model, val_loader)
        acc = (y_true == y_pred).mean()

        print(f"\nEpoch {epoch:02d} | train_loss={total_loss/len(tr_ds):.4f} | val_acc={acc:.3f}")
        print(confusion_matrix(y_true, y_pred))
        print(classification_report(y_true, y_pred, target_names=LABELS, zero_division=0))

        if acc > best_acc:
            best_acc = acc
            torch.save({"state": model.state_dict(), "input_dim": input_dim, "labels": LABELS}, "models/judo_pose_gru.pt")
            print("  → saved models/judo_pose_gru.pt")


if __name__ == "__main__":
    main()
