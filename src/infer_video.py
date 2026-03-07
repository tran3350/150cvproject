"""
Step 5: End-to-end pose-based inference on a video.

Pipeline:
  1. YOLOv8-pose + ByteTrack  →  raw keypoints per frame
  2. Normalize keypoints       →  (n_frames, 2, 17, 3)
  3. Sliding window features   →  (N, 64, 175)
  4. PoseGRU classifier        →  softmax probs per window
  5. Temporal smoothing        →  event extraction

Outputs:
  outputs/pose_preds.csv    — per-window probabilities
  outputs/pose_events.json  — detected technique events with timestamps
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LABELS: list[str] = []   # loaded from models/judo_pose_gru.pt at runtime

# Normalization landmarks (COCO-17)
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP,      R_HIP      = 11, 12

INTER_JOINTS  = [9, 10, 5, 6, 11]
FPS_USED      = 30
WIN_FRAMES    = 64
STRIDE_FRAMES = 12


# ---------- model (must match train_classifier.py) ---------------------------

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
        _, h = self.gru(x)
        return self.head(self.drop(h[-1]))


# ---------- pose helpers (must match pose_estimate.py / build_features.py) --

def normalize_person(kp17: np.ndarray) -> np.ndarray:
    kp = kp17.copy().astype(np.float32)
    hip_mid   = (kp[L_HIP, :2]      + kp[R_HIP,      :2]) / 2.0
    neck      = (kp[L_SHOULDER, :2] + kp[R_SHOULDER,  :2]) / 2.0
    torso_len = float(np.linalg.norm(neck - hip_mid)) + 1e-6
    kp[:, :2] = (kp[:, :2] - hip_mid) / torso_len
    return kp


def frame_features(kp: np.ndarray, kp_prev: np.ndarray) -> np.ndarray:
    """kp, kp_prev : (2, 17, 3)  →  (175,) float32"""
    xy    = kp[:, :, :2].reshape(-1)
    conf  = kp[:, :,  2].reshape(-1)
    vel   = (kp[:, :, :2] - kp_prev[:, :, :2]).reshape(-1)
    inter = np.array(
        [np.linalg.norm(kp[0, j, :2] - kp[1, j, :2]) for j in INTER_JOINTS],
        dtype=np.float32,
    )
    return np.concatenate([xy, conf, vel, inter]).astype(np.float32)


# ---------- tracking / pose --------------------------------------------------

def run_pose_tracking(video_path: Path, conf: float = 0.35):
    """Returns list of (frame_idx, track_id, kp17 (17,3))."""
    model      = YOLO("yolov8m-pose.pt")
    detections = []

    results = model.track(
        source=str(video_path),
        classes=[0],
        conf=conf,
        tracker="bytetrack.yaml",
        stream=True,
        verbose=False,
        device=0,
    )

    for frame_idx, r in enumerate(results):
        if r.boxes is None or r.boxes.id is None or r.keypoints is None:
            continue
        kps = r.keypoints.data.cpu().numpy()
        ids = r.boxes.id.int().cpu().numpy()
        for i in range(len(ids)):
            detections.append((frame_idx, int(ids[i]), kps[i]))

        if frame_idx % 500 == 0:
            print(f"  frame {frame_idx:5d}")

    return detections


def detections_to_keypoints(detections) -> np.ndarray:
    """Returns (n_frames, 2, 17, 3) float32 with normalized keypoints."""
    if not detections:
        return np.zeros((0, 2, 17, 3), dtype=np.float32)

    n_frames = max(d[0] for d in detections) + 1
    counts   = Counter(d[1] for d in detections)
    top2     = [tid for tid, _ in counts.most_common(2)]
    print(f"Top-2 track IDs: {top2}")

    kp_arr = np.zeros((n_frames, 2, 17, 3), dtype=np.float32)
    for fid, tid, kp17 in detections:
        if tid in top2:
            kp_arr[fid, top2.index(tid)] = normalize_person(kp17)

    return kp_arr


# ---------- post-processing --------------------------------------------------

def smooth_probs(probs: np.ndarray, k: int = 5) -> np.ndarray:
    if k <= 1:
        return probs
    pad = k // 2
    out = np.zeros_like(probs)
    for c in range(probs.shape[1]):
        xpad      = np.pad(probs[:, c], (pad, pad), mode="edge")
        out[:, c] = np.convolve(xpad, np.ones(k) / k, mode="valid")
    return out


def probs_to_events(probs, starts_s, label_idx, thresh=0.6, min_dur=0.5):
    events, active, s0, peak_p, peak_t = [], False, None, 0.0, None
    for p, t in zip(probs, starts_s):
        if not active and p >= thresh:
            active, s0, peak_p, peak_t = True, t, p, t
        elif active:
            if p > peak_p:
                peak_p, peak_t = p, t
            if p < thresh:
                if t - s0 >= min_dur:
                    events.append({
                        "label":     LABELS[label_idx],
                        "start":     float(s0),
                        "end":       float(t),
                        "peak_time": float(peak_t),
                        "peak_prob": float(peak_p),
                    })
                active = False
    if active and starts_s[-1] - s0 >= min_dur:
        events.append({
            "label":     LABELS[label_idx],
            "start":     float(s0),
            "end":       float(starts_s[-1]),
            "peak_time": float(peak_t),
            "peak_prob": float(peak_p),
        })
    return events


# ---------- main -------------------------------------------------------------

def main():
    video_path = Path("data/raw_videos/3d3ad63e-trainingdata.mp4")
    model_path = Path("models/judo_pose_gru.pt")
    out_dir    = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    # 1. Track + pose
    print("Running YOLOv8-pose + ByteTrack...")
    detections = run_pose_tracking(video_path)
    kp         = detections_to_keypoints(detections)   # (n_frames, 2, 17, 3)
    n_frames   = kp.shape[0]
    print(f"Keypoints shape: {kp.shape}")

    # 2. Load GRU model
    global LABELS
    ckpt      = torch.load(model_path, map_location=DEVICE)
    input_dim = ckpt["input_dim"]
    LABELS    = ckpt["labels"]
    model     = PoseGRU(input_dim, n_classes=len(LABELS)).to(DEVICE)
    model.load_state_dict(ckpt["state"])
    model.eval()

    # 3. Sliding-window inference
    all_probs, all_starts = [], []
    with torch.no_grad():
        for start_f in range(0, n_frames - WIN_FRAMES + 1, STRIDE_FRAMES):
            clip  = kp[start_f:start_f + WIN_FRAMES]          # (64, 2, 17, 3)
            feats = []
            for t in range(WIN_FRAMES):
                prev = clip[t - 1] if t > 0 else clip[t]
                feats.append(frame_features(clip[t], prev))
            x      = torch.from_numpy(np.stack(feats)).unsqueeze(0).to(DEVICE)  # (1, 64, 175)
            logits = model(x)
            prob   = torch.softmax(logits, dim=1).cpu().numpy()[0]
            all_probs.append(prob)
            all_starts.append(start_f / FPS_USED)

    probs    = smooth_probs(np.stack(all_probs), k=5)   # (N, 4)
    starts_s = np.array(all_starts, dtype=np.float32)

    # 4. Save per-window CSV
    csv_path = out_dir / "pose_preds.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("start_s," + ",".join(f"p_{l}" for l in LABELS) + ",pred\n")
        for t, p in zip(starts_s, probs):
            pred = LABELS[int(np.argmax(p))]
            f.write(f"{t:.3f}," + ",".join(f"{v:.5f}" for v in p) + f",{pred}\n")

    # 5. Extract and save events
    events = []
    for li in range(1, len(LABELS)):
        events.extend(probs_to_events(probs[:, li], starts_s, li, thresh=0.6, min_dur=0.5))

    events = sorted(events, key=lambda e: e["start"])
    (out_dir / "pose_events.json").write_text(json.dumps(events, indent=2), encoding="utf-8")

    print(f"\nWrote {csv_path} and outputs/pose_events.json")
    print(f"Detected {len(events)} event(s):")
    for e in events[:10]:
        print(f"  {e['label']:6s}  {e['start']:.1f}s – {e['end']:.1f}s  peak={e['peak_prob']:.2f}")


if __name__ == "__main__":
    main()
