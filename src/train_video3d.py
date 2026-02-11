import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LABELS = ["none", "seoi", "nage", "ippon"]

class VideoWindows(Dataset):
    def __init__(self, x_npz, y_npy):
        self.X = np.load(x_npz)["X"]  # (N, T, H, W, C) uint8
        self.y = np.load(y_npy)       # (N,) int64

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]  # uint8
        y = int(self.y[idx])

        # normalize to float32 [0,1]
        x = torch.from_numpy(x).float() / 255.0  # (T,H,W,C)
        # rearrange to (C,T,H,W) for 3D conv
        x = x.permute(3, 0, 1, 2)  # (C,T,H,W)

        return x, torch.tensor(y, dtype=torch.long)

class Small3DCNN(nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=(3,5,5), stride=(1,2,2), padding=(1,2,2)),
            nn.BatchNorm3d(16),
            nn.ReLU(),

            nn.Conv3d(16, 32, kernel_size=(3,3,3), stride=(1,2,2), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),

            nn.Conv3d(32, 64, kernel_size=(3,3,3), stride=(2,2,2), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),

            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        self.classifier = nn.Linear(64, n_classes)

    def forward(self, x):
        z = self.features(x)           # (B,64,1,1,1)
        z = z.flatten(1)               # (B,64)
        return self.classifier(z)      # (B,n_classes)

def eval_model(model, loader):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            logits = model(x)
            pred = logits.argmax(dim=1).cpu().numpy()
            ys.append(y.numpy())
            ps.append(pred)
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    return y_true, y_pred

def main():
    X_train = "data/windows/X_train.npz"
    y_train = "data/windows/y_train.npy"
    X_val   = "data/windows/X_val.npz"
    y_val   = "data/windows/y_val.npy"

    train_ds = VideoWindows(X_train, y_train)
    val_ds   = VideoWindows(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)

    # class weights (helps because seoi is smaller)
    y = np.load(y_train)
    counts = np.bincount(y, minlength=len(LABELS)).astype(np.float32)
    weights = (counts.sum() / np.maximum(counts, 1.0))
    weights = torch.tensor(weights, device=DEVICE)

    model = Small3DCNN(n_classes=len(LABELS)).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)

    best = 0.0
    os.makedirs("models", exist_ok=True)

    for epoch in range(1, 21):
        model.train()
        total_loss = 0.0
        for x, yb in train_loader:
            x = x.to(DEVICE)
            yb = yb.to(DEVICE)

            optim.zero_grad()
            logits = model(x)
            loss = criterion(logits, yb)
            loss.backward()
            optim.step()
            total_loss += loss.item() * x.size(0)

        y_true, y_pred = eval_model(model, val_loader)
        acc = (y_true == y_pred).mean()
        print(f"Epoch {epoch:02d} | train_loss={total_loss/len(train_ds):.4f} | val_acc={acc:.3f}")
        print(confusion_matrix(y_true, y_pred))
        print(classification_report(y_true, y_pred, target_names=LABELS, zero_division=0))

        if acc > best:
            best = acc
            torch.save({"state": model.state_dict()}, "models/judo_3dcnn.pt")
            print("saved models/judo_3dcnn.pt")

if __name__ == "__main__":
    main()
