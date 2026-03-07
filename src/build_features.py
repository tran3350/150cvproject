"""
Step 3: Build sliding-window features from compilation keypoints.

Labels come from folder names (data/poses/{label}/*_kp.npy).
No segments.json needed.

Motion energy filter:
  Each window gets a motion score = mean absolute joint velocity across
  all confident keypoints. Windows below MOTION_THRESH are relabeled "none"
  — these are the between-throw gaps in compilation videos.

Feature vector: 175 values per frame
  xy   2*17*2 = 68   normalized (x, y) for both athletes
  conf 2*17   = 34   keypoint confidence
  vel  2*17*2 = 68   frame-to-frame delta
  inter       =  5   inter-athlete distances at key joints
               ----
               175

Saves: data/pose_windows/{X_train,y_train,X_val,y_val}.npy + meta.json
"""
import json
import math
from pathlib import Path

import numpy as np

POSES_DIR     = Path("data/poses")
OUT_DIR       = Path("data/pose_windows")

FPS           = 30
WIN_FRAMES    = 64
STRIDE_FRAMES = 12
VAL_FRACTION  = 0.2
SEED          = 1337

# "none" must be first so its ID is 0
LABELS      = ["none", "seoi"]
LABEL_TO_ID = {l: i for i, l in enumerate(LABELS)}

INTER_JOINTS   = [9, 10, 5, 6, 11]   # l_wrist, r_wrist, l_shoulder, r_shoulder, l_hip
MOTION_THRESH  = 0.08                 # tune up if too much filler leaks through,
                                      # tune down if too many throws get dropped


# ---------- features ---------------------------------------------------------

def frame_features(kp: np.ndarray, kp_prev: np.ndarray) -> np.ndarray:
    """kp, kp_prev : (2, 17, 3) → (175,) float32"""
    xy    = kp[:, :, :2].reshape(-1)
    conf  = kp[:, :,  2].reshape(-1)
    vel   = (kp[:, :, :2] - kp_prev[:, :, :2]).reshape(-1)
    inter = np.array(
        [np.linalg.norm(kp[0, j, :2] - kp[1, j, :2]) for j in INTER_JOINTS],
        dtype=np.float32,
    )
    return np.concatenate([xy, conf, vel, inter]).astype(np.float32)


def motion_score(clip: np.ndarray) -> float:
    """
    clip : (T, 2, 17, 3)
    Returns mean absolute velocity across all confident joints.
    High score = fast movement = likely a throw.
    """
    vel  = np.abs(np.diff(clip[:, :, :, :2], axis=0))  # (T-1, 2, 17, 2)
    conf = clip[:-1, :, :, 2]                           # (T-1, 2, 17)
    mask = conf > 0.3
    if mask.sum() == 0:
        return 0.0
    # expand mask to cover both x and y
    mask2 = np.stack([mask, mask], axis=-1)              # (T-1, 2, 17, 2)
    return float(vel[mask2].mean())


# ---------- split ------------------------------------------------------------

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


# ---------- main -------------------------------------------------------------

def main():
    kp_paths = sorted(POSES_DIR.rglob("*_kp.npy"))
    if not kp_paths:
        raise FileNotFoundError(
            f"No keypoint files found under {POSES_DIR.resolve()}. "
            "Run pose_estimate.py first."
        )

    all_windows, all_labels = [], []
    label_counts = {l: 0 for l in LABELS}

    for kp_path in kp_paths:
        label = kp_path.parent.name
        if label not in LABEL_TO_ID:
            print(f"Skipping {kp_path} — unknown label '{label}'")
            continue

        folder_label_id = LABEL_TO_ID[label]
        none_id         = LABEL_TO_ID["none"]

        kp       = np.load(kp_path)    # (n_frames, 2, 17, 3)
        n_frames = kp.shape[0]

        n_active = 0
        n_none   = 0

        for start_f in range(0, n_frames - WIN_FRAMES + 1, STRIDE_FRAMES):
            clip = kp[start_f:start_f + WIN_FRAMES]    # (64, 2, 17, 3)

            score        = motion_score(clip)
            win_label_id = folder_label_id if score >= MOTION_THRESH else none_id

            feats = []
            for t in range(WIN_FRAMES):
                prev = clip[t - 1] if t > 0 else clip[t]
                feats.append(frame_features(clip[t], prev))

            all_windows.append(np.stack(feats, axis=0))    # (64, 175)
            all_labels.append(win_label_id)

            if win_label_id == folder_label_id:
                n_active += 1
            else:
                n_none += 1

        label_counts[label] += n_active
        label_counts["none"] += n_none
        print(f"  {kp_path.stem}: active={n_active}  none(filtered)={n_none}  score_thresh={MOTION_THRESH}")

    X = np.stack(all_windows, axis=0)           # (N, 64, 175)
    y = np.array(all_labels,  dtype=np.int64)   # (N,)

    print(f"\nTotal windows : {len(X)}")
    print(f"Label counts  : {label_counts}")

    X_tr, y_tr, X_val, y_val = stratified_split(X, y, VAL_FRACTION, SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    np.save(OUT_DIR / "X_train.npy", X_tr)
    np.save(OUT_DIR / "y_train.npy", y_tr)
    np.save(OUT_DIR / "X_val.npy",   X_val)
    np.save(OUT_DIR / "y_val.npy",   y_val)

    meta = {
        "labels":        LABELS,
        "feature_dim":   int(X.shape[-1]),
        "win_frames":    WIN_FRAMES,
        "fps":           FPS,
        "motion_thresh": MOTION_THRESH,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved to {OUT_DIR.resolve()}")
    print(f"\nIf active count is too low, decrease MOTION_THRESH (currently {MOTION_THRESH})")
    print(f"If too much filler leaks in,  increase MOTION_THRESH")


if __name__ == "__main__":
    main()
