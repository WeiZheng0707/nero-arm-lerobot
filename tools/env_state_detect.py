"""Generate reviewable env-state labels from captured base-camera frames.

The raw episode is never modified. Labels are written to env_state.jsonl and
optional overlays are written under review/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from lerobot.nero_tools.color_detector import detect_scene


def parse_roi(value: str | None):
    if value is None:
        return None
    parts = [int(p) for p in value.replace(" ", "").split(",")]
    if len(parts) != 4 or parts[2] <= 0 or parts[3] <= 0:
        raise argparse.ArgumentTypeError("ROI must be x,y,w,h with positive w,h")
    return tuple(parts)


def draw_scene(image, scene):
    height, width = image.shape[:2]
    colors = {"cube": (0, 0, 255), "ball": (255, 0, 0), "bag": (0, 180, 180)}
    for name, box in scene.items():
        # scene may also carry diagnostic keys such as ball_rejected (a string).
        if not isinstance(box, dict) or not box.get("visible"):
            continue
        x = int(box["x"] * width)
        y = int(box["y"] * height)
        w = int(box["w"] * width)
        h = int(box["h"] * height)
        cv2.rectangle(image, (x, y), (x + w, y + h), colors[name], 2)
        cv2.putText(image, f"{name} {box['conf']:.2f}", (x, max(15, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[name], 1, cv2.LINE_AA)
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("episode", type=Path)
    parser.add_argument("--base-roi", type=parse_roi, default=(0, 100, 520, 380))
    parser.add_argument("--ball-color", default="none")
    parser.add_argument("--cube-color", default="red")
    parser.add_argument("--morphology-kernel", type=int, default=5)
    parser.add_argument("--ball-min-sat", type=int, default=110,
                        help="Blue saturation floor for the ball. The base view's pale floor "
                             "clears the wrist default of 30 and reads as a ball.")
    parser.add_argument("--ball-min-val", type=int, default=70)
    parser.add_argument("--review-every", type=int, default=30)
    args = parser.parse_args()
    if args.morphology_kernel < 1 or args.morphology_kernel % 2 == 0:
        parser.error("--morphology-kernel must be a positive odd integer")
    episode = args.episode
    rows = [json.loads(line) for line in (episode / "timestamps.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    output = episode / "env_state.jsonl"
    review = episode / "review"
    review.mkdir(exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            image_path = episode / row["cameras"]["base"]["image"]
            image = cv2.imread(str(image_path))
            if image is None:
                raise RuntimeError(f"Cannot read {image_path}")
            scene = detect_scene(
                image, args.cube_color,
                ball_color=None if args.ball_color.lower() == "none" else args.ball_color,
                roi=args.base_roi, exclude_top_ratio=0.0,
                morphology_kernel=args.morphology_kernel,
                ball_min_sat=args.ball_min_sat, ball_min_val=args.ball_min_val,
            )
            row_out = {"frame_index": index, "host_time_s": row["host_time_s"]}
            for name in ("cube", "ball", "bag"):
                row_out[name] = scene.get(name) or {}
            handle.write(json.dumps(row_out) + "\n")
            if index % args.review_every == 0:
                cv2.imwrite(str(review / f"frame_{index:06d}.jpg"), draw_scene(image, scene))
    print(f"WROTE {output}")
    print(f"REVIEW {review} (every {args.review_every} frames)")


if __name__ == "__main__":
    main()
