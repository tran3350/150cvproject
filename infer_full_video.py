import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LABELS = ["none", "seoi", "nage", "ippon"]

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
        z = self.features(x)
        z = z.flatten(1)
        return self.classifier(z)

def read_video_sampled(video_path: Path, fps_target: int, resize_wh=(224, 224)):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps_src = cap.get(cv2.CAP_PROP_FPS)
    if fps_src <= 0:
        fps_src = fps_target

    step = fps_src / fps_target
    frames = []
    next_pick = 0.0
    i = 0

    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        if i + 1e-9 >= next_pick:
            bgr = cv2.resize(bgr, resize_wh, interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(rgb)
            next_pick += step
        i += 1

    cap.release()
    return np.stack(frames, axis=0).astype(np.uint8)

def sliding_windows(frames, win_frames, stride_frames):
    T = len(frames)
    for start in range(0, T - win_frames + 1, stride_frames):
        yield start, frames[start:start+win_frames]

def smooth_probs(probs, k=5):
    # probs: (N, C)
    if k <= 1:
        return probs
    pad = k // 2
    out = np.zeros_like(probs, dtype=np.float32)
    for c in range(probs.shape[1]):
        x = probs[:, c]
        xpad = np.pad(x, (pad, pad), mode="edge")
        out[:, c] = np.convolve(xpad, np.ones(k)/k, mode="valid")
    return out

def probs_to_events(probs, starts_s, label_idx, thresh=0.6, min_duration_s=0.5):
    """
    Convert a single label probability curve into event segments.
    probs: (N,) for that label
    starts_s: (N,) window start times in seconds
    """
    events = []
    active = False
    s0 = None
    peak_p = 0.0
    peak_t = None

    for p, t in zip(probs, starts_s):
        if not active and p >= thresh:
            active = True
            s0 = t
            peak_p = p
            peak_t = t
        elif active:
            if p > peak_p:
                peak_p = p
                peak_t = t
            if p < thresh:
                e0 = t
                if e0 - s0 >= min_duration_s:
                    events.append({
                        "label": LABELS[label_idx],
                        "start": float(s0),
                        "end": float(e0),
                        "peak_time": float(peak_t),
                        "peak_prob": float(peak_p)
                    })
                active = False
                s0 = None

    # close event if it ends at the last window
    if active:
        e0 = starts_s[-1]
        if e0 - s0 >= min_duration_s:
            events.append({
                "label": LABELS[label_idx],
                "start": float(s0),
                "end": float(e0),
                "peak_time": float(peak_t),
                "peak_prob": float(peak_p)
            })
    return events

def main():
    video_path = Path("data/raw_videos/3d3ad63e-trainingdata.mp4")  # change as needed
    model_path = Path("models/judo_3dcnn.pt")

    fps_used = 30
    win_frames = 64
    stride_frames = 12
    resize_wh = (224, 224)

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    frames = read_video_sampled(video_path, fps_used, resize_wh=resize_wh)
    print(f"Frames sampled: {len(frames)}")

    model = Small3DCNN(n_classes=len(LABELS)).to(DEVICE)
    ckpt = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(ckpt["state"])
    model.eval()

    all_probs = []
    all_starts = []

    with torch.no_grad():
        for start_f, clip in sliding_windows(frames, win_frames, stride_frames):
            x = torch.from_numpy(clip).float() / 255.0  # (T,H,W,C)
            x = x.permute(3, 0, 1, 2).unsqueeze(0).to(DEVICE)  # (1,C,T,H,W)
            logits = model(x)
            prob = torch.softmax(logits, dim=1).cpu().numpy()[0]  # (C,)
            all_probs.append(prob)
            all_starts.append(start_f / fps_used)

    probs = np.stack(all_probs, axis=0)   # (N, C)
    starts_s = np.array(all_starts, dtype=np.float32)  # (N,)

    probs_s = smooth_probs(probs, k=5)

    # Save per-window predictions
    csv_path = out_dir / "preds.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("start_s," + ",".join([f"p_{l}" for l in LABELS]) + ",pred\n")
        for t, p in zip(starts_s, probs_s):
            pred = int(np.argmax(p))
            f.write(f"{t:.3f}," + ",".join([f"{x:.5f}" for x in p]) + f",{LABELS[pred]}\n")

    # Extract events for non-none classes
    events = []
    for li in range(1, len(LABELS)):
        events.extend(probs_to_events(probs_s[:, li], starts_s, li, thresh=0.6, min_duration_s=0.5))

    events = sorted(events, key=lambda e: e["start"])
    (out_dir / "events.json").write_text(json.dumps(events, indent=2), encoding="utf-8")

    print(f"Wrote {csv_path} and outputs/events.json")
    print("Top events (first 10):")
    for e in events[:10]:
        print(e)

if __name__ == "__main__":
    main()
