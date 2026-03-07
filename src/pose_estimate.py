"""
Step 2: Convert raw detections into clean, normalized keypoint arrays.

For each video, selects the 2 athletes per frame by LARGEST BOUNDING BOX AREA
(not by global track-ID frequency). This is robust to compilation videos where
different athlete pairs appear in each clip — the athletes are always the
largest people in frame; referees and crowd are smaller/further away.

Normalization: origin = hip midpoint, scale = torso length.

Saves per video:
  data/poses/{label}/{video_stem}_kp.npy  — (n_frames, 2, 17, 3) float32
"""
from pathlib import Path
import numpy as np

POSES_DIR = Path("data/poses")

L_SHOULDER, R_SHOULDER = 5, 6
L_HIP,      R_HIP      = 11, 12


def normalize_person(kp17: np.ndarray) -> np.ndarray:
    kp = kp17.copy().astype(np.float32)
    hip_mid   = (kp[L_HIP, :2]      + kp[R_HIP,      :2]) / 2.0
    neck      = (kp[L_SHOULDER, :2] + kp[R_SHOULDER,  :2]) / 2.0
    torso_len = float(np.linalg.norm(neck - hip_mid)) + 1e-6
    kp[:, :2] = (kp[:, :2] - hip_mid) / torso_len
    return kp


def bbox_area(bbox: np.ndarray) -> float:
    return float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))


def process_raw(raw_path: Path, kp_out_path: Path):
    data      = np.load(raw_path)
    frame_ids = data["frame_ids"]   # (N,)
    bboxes    = data["bboxes"]      # (N, 4)
    keypoints = data["keypoints"]   # (N, 17, 3)

    n_frames = int(frame_ids.max()) + 1
    kp_out   = np.zeros((n_frames, 2, 17, 3), dtype=np.float32)

    # Group detection indices by frame
    frame_to_dets: dict[int, list[int]] = {}
    for i, fid in enumerate(frame_ids):
        frame_to_dets.setdefault(int(fid), []).append(i)

    for fid, det_indices in frame_to_dets.items():
        # Sort by bbox area descending → top-2 are the athletes
        det_indices_by_area = sorted(
            det_indices, key=lambda i: bbox_area(bboxes[i]), reverse=True
        )
        for slot, det_i in enumerate(det_indices_by_area[:2]):
            kp_out[fid, slot] = normalize_person(keypoints[det_i])

    np.save(kp_out_path, kp_out)

    hip_vis = [(kp_out[:, s, L_HIP, 2] > 0).mean() for s in range(2)]
    print(f"  {raw_path.stem}: frames={n_frames} | hip_vis=[{hip_vis[0]:.1%}, {hip_vis[1]:.1%}]")


def main():
    raw_paths = sorted(POSES_DIR.rglob("*_raw.npz"))
    if not raw_paths:
        raise FileNotFoundError(
            f"No raw pose files found under {POSES_DIR.resolve()}. "
            "Run track_people.py first."
        )

    for raw_path in raw_paths:
        kp_out_path = raw_path.with_name(raw_path.stem.replace("_raw", "_kp") + ".npy")
        if kp_out_path.exists():
            print(f"Skipping {raw_path.stem} (already processed)")
            continue
        print(f"Processing {raw_path.stem} ...")
        process_raw(raw_path, kp_out_path)

    print("\nDone. Next: python src/build_features.py")


if __name__ == "__main__":
    main()
