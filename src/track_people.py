"""
Step 1: Detect and track people using YOLOv8-pose + ByteTrack.

Walks data/compilations/{label}/*.mp4 and processes every video.
Skips videos whose output already exists (safe to re-run).

Saves per video:
  data/poses/{label}/{video_stem}_raw.npz
    frame_ids  : (N,)       int32
    track_ids  : (N,)       int32
    bboxes     : (N, 4)     float32  xyxy pixels
    keypoints  : (N, 17, 3) float32  COCO-17 (x_px, y_px, conf)
"""
from pathlib import Path
import numpy as np
from ultralytics import YOLO

COMPILATIONS_DIR = Path("data/compilations")
POSES_DIR        = Path("data/poses")

MODEL_NAME = "yolov8m-pose.pt"
CONF       = 0.35


def process_video(model, video_path: Path, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frame_ids_list = []
    track_ids_list = []
    bboxes_list    = []
    keypoints_list = []

    results = model.track(
        source=str(video_path),
        classes=[0],
        conf=CONF,
        tracker="bytetrack.yaml",
        stream=True,
        verbose=False,
        device=0,
    )

    for frame_idx, r in enumerate(results):
        if r.boxes is None or r.boxes.id is None or r.keypoints is None:
            continue

        kps = r.keypoints.data.cpu().numpy()   # (n_det, 17, 3)
        ids = r.boxes.id.int().cpu().numpy()    # (n_det,)
        bbs = r.boxes.xyxy.cpu().numpy()        # (n_det, 4)

        for i in range(len(ids)):
            frame_ids_list.append(frame_idx)
            track_ids_list.append(int(ids[i]))
            bboxes_list.append(bbs[i])
            keypoints_list.append(kps[i])

        if frame_idx % 500 == 0:
            print(f"    frame {frame_idx:5d}  detections so far: {len(frame_ids_list)}")

    if not frame_ids_list:
        print(f"  No detections found, skipping save.")
        return

    np.savez_compressed(
        out_path,
        frame_ids=np.array(frame_ids_list, dtype=np.int32),
        track_ids=np.array(track_ids_list, dtype=np.int32),
        bboxes=np.array(bboxes_list,       dtype=np.float32),
        keypoints=np.array(keypoints_list, dtype=np.float32),
    )
    print(f"  Saved {len(frame_ids_list)} detections → {out_path}")


def main():
    video_paths = sorted(COMPILATIONS_DIR.rglob("*.mp4"))
    if not video_paths:
        raise FileNotFoundError(
            f"No .mp4 files found under {COMPILATIONS_DIR.resolve()}. "
            "Run download_videos.py first."
        )

    model = YOLO(MODEL_NAME)

    for video_path in video_paths:
        label    = video_path.parent.name
        out_path = POSES_DIR / label / f"{video_path.stem}_raw.npz"

        if out_path.exists():
            print(f"Skipping {video_path.name} (already processed)")
            continue

        print(f"\n[{label}] {video_path.name}")
        process_video(model, video_path, out_path)

    print("\nDone. Next: python src/pose_estimate.py")


if __name__ == "__main__":
    main()
