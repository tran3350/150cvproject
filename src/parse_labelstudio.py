import json
import os
from pathlib import Path

def parse_labelstudio(export_path: str, out_path: str):
    export_path = Path(export_path)
    out_path = Path(out_path)

    with export_path.open("r", encoding="utf-8") as f:
        tasks = json.load(f)

    # Output format:
    # {
    #   "trainingdata.mp4": [
    #       {"start": 86.0, "end": 89.0, "label": "seoi"},
    #       ...
    #   ]
    # }
    out = {}

    for task in tasks:
        video_field = task.get("data", {}).get("video")
        if not video_field:
            continue

        # Example: "/data/upload/1/3d3ad63e-trainingdata.mp4"
        video_name = Path(video_field).name

        segments = []
        for ann in task.get("annotations", []):
            for r in ann.get("result", []):
                if r.get("type") != "timelinelabels":
                    continue
                value = r.get("value", {})
                labels = value.get("timelinelabels", [])
                ranges = value.get("ranges", [])
                if not labels or not ranges:
                    continue

                label = labels[0]  # you used single label per region
                for seg in ranges:
                    FPS = 29.98  # match your printed src_fps (or use 30.0)

                    start = float(seg["start"]) / FPS
                    end   = float(seg["end"]) / FPS

                    if end <= start:
                        continue
                    segments.append({"start": start, "end": end, "label": label})

        # Sort by start time
        segments.sort(key=lambda s: (s["start"], s["end"]))

        out[video_name] = segments

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote {out_path} with {sum(len(v) for v in out.values())} segments across {len(out)} video(s).")

if __name__ == "__main__":
    parse_labelstudio(
        export_path="data/labels/labelstudio_export.json",
        out_path="data/labels/segments.json",
    )
