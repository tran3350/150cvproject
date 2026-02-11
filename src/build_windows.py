import json
import math
from pathlib import Path
import cv2
import numpy as np

# ---------- Config ----------
VIDEO_PATH = Path("data/raw_videos/3d3ad63e-trainingdata.mp4")  # MUST match your file name
SEGMENTS_PATH = Path("data/labels/segments.json")

OUT_DIR = Path("data/windows")
FPS_TARGET = 30
RESIZE_W, RESIZE_H = 224, 224
WIN_FRAMES = 64
STRIDE_FRAMES = 12
VAL_FRACTION = 0.2
SEED = 1337

LABELS = ["none", "seoi", "nage", "ippon"]
LABEL_TO_ID = {name: i for i, name in enumerate(LABELS)}

# ---------- Helpers ----------
def interval_overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))

def assign_label_for_window(win_start_s, win_end_s, segments, min_overlap_ratio=0.05):
    win_len = win_end_s - win_start_s
    best_label = "none"
    best_ov = 0.0
    for seg in segments:
        ov = interval_overlap(win_start_s, win_end_s, seg["start"], seg["end"])
        if ov > best_ov:
            best_ov = ov
            best_label = seg["label"]
    if best_ov < min_overlap_ratio * win_len:
        return "none"
    return best_label if best_label in LABEL_TO_ID else "none"

def read_video_frames(video_path: Path, fps_target: int, resize_wh=(224, 224)):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps_src = cap.get(cv2.CAP_PROP_FPS)
    if fps_src <= 0:
        fps_src = 60.0
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
    return np.stack(frames, axis=0).astype(np.uint8), fps_src

def make_windows(frames, segments, fps_used, win_frames, stride_frames):
    T = frames.shape[0]
    windows, labels, meta = [], [], []
    for start_f in range(0, T - win_frames + 1, stride_frames):
        end_f = start_f + win_frames
        win = frames[start_f:end_f]
        win_start_s = start_f / fps_used
        win_end_s = end_f / fps_used
        label = assign_label_for_window(win_start_s, win_end_s, segments)
        windows.append(win)
        labels.append(LABEL_TO_ID[label])
        meta.append({"start_s": win_start_s, "end_s": win_end_s, "label": label})
    return np.stack(windows, axis=0), np.array(labels, dtype=np.int64), meta

def stratified_split(X, y, val_fraction, seed):
    rng = np.random.default_rng(seed)
    idx_by_class = {}
    for i, cls in enumerate(y):
        idx_by_class.setdefault(int(cls), []).append(i)

    train_idx, val_idx = [], []
    for cls, idxs in idx_by_class.items():
        idxs = np.array(idxs, dtype=np.int64)
        rng.shuffle(idxs)
        n_val = int(math.ceil(len(idxs) * val_fraction))
        val_idx.extend(idxs[:n_val].tolist())
        train_idx.extend(idxs[n_val:].tolist())

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]

def main():
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH.resolve()}")

    if not SEGMENTS_PATH.exists():
        raise FileNotFoundError(f"Segments not found: {SEGMENTS_PATH.resolve()} (run parse_labelstudio first)")

    with SEGMENTS_PATH.open("r", encoding="utf-8") as f:
        segmap = json.load(f)

    video_key = VIDEO_PATH.name
    if video_key not in segmap:
        raise KeyError(f"Video '{video_key}' not found in segments.json keys: {list(segmap.keys())[:5]}")

    segments = segmap[video_key]
    segments = sorted(segments, key=lambda s: (s["start"], s["end"]))

    frames, fps_src = read_video_frames(VIDEO_PATH, FPS_TARGET, resize_wh=(RESIZE_W, RESIZE_H))
    fps_used = FPS_TARGET

    print(f"Video: {VIDEO_PATH} | src_fps={fps_src:.2f} | sampled_fps={fps_used} | frames={len(frames)}")
    print(f"Segments: {len(segments)}")

    X, y, meta = make_windows(frames, segments, fps_used, WIN_FRAMES, STRIDE_FRAMES)
    counts = np.bincount(y, minlength=len(LABELS))
    print(f"Windows: {len(X)} | X={X.shape} | y_counts={dict(zip(LABELS, counts.tolist()))}")

    X_train, y_train, X_val, y_val = stratified_split(X, y, VAL_FRACTION, SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(OUT_DIR / "X_train.npz", X=X_train)
    np.save(OUT_DIR / "y_train.npy", y_train)
    np.savez_compressed(OUT_DIR / "X_val.npz", X=X_val)
    np.save(OUT_DIR / "y_val.npy", y_val)

    with (OUT_DIR / "meta.json").open("w", encoding="utf-8") as f:
        json.dump({"labels": LABELS, "windows": meta}, f, indent=2)

    print(f"Saved windows to: {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()

