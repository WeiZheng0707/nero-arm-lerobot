"""Review raw eye-in-hand episodes and save human success labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2


LABELS = {ord("y"): "success", ord("n"): "failure", ord("u"): "uncertain"}


def load_reviews(path: Path) -> dict[tuple[str, str], str]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return {
            (row["dataset"], row["episode"]): row.get("label", "")
            for row in csv.DictReader(stream)
        }


def save_reviews(path: Path, reviews: dict[tuple[str, str], str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["dataset", "episode", "label"])
        for (dataset, episode), label in sorted(reviews.items()):
            writer.writerow([dataset, episode, label])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True,
                        help="Raw dataset path; pass once for red and once for blue")
    parser.add_argument("--output", type=Path, default=Path("outputs/eye_in_hand_review.csv"))
    parser.add_argument("--samples", type=int, default=12,
                        help="Number of evenly spaced frames shown per episode")
    args = parser.parse_args()
    if args.samples < 2:
        parser.error("--samples must be at least 2")

    episodes = [
        (source.name, episode)
        for source in args.source
        for episode in sorted(source.glob("episode_*"))
    ]
    reviews = load_reviews(args.output)
    window = "Eye-in-hand episode review"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    try:
        for dataset, episode in episodes:
            metadata = json.loads((episode / "metadata.json").read_text(encoding="utf-8"))
            rows = [json.loads(line) for line in
                    (episode / "timestamps.jsonl").read_text(encoding="utf-8").splitlines()
                    if line.strip()]
            if not rows:
                raise ValueError(f"No frames in {episode}")
            indices = sorted({round(i * (len(rows) - 1) / (args.samples - 1))
                              for i in range(args.samples)})
            label = reviews.get((dataset, episode.name), "")
            for position, index in enumerate(indices, 1):
                frame_path = episode / rows[index]["image"]
                frame = cv2.imread(str(frame_path))
                if frame is None:
                    raise ValueError(f"Cannot read image: {frame_path}")
                cv2.putText(frame, f"{dataset}/{episode.name}  {position}/{len(indices)}",
                            (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                cv2.putText(frame, f"Current label: {label or 'unreviewed'}  result={metadata.get('result')}",
                            (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                cv2.putText(frame, "Y=success  N=failure  U=uncertain  Space=next frame  Q=save and quit",
                            (12, frame.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 2)
                cv2.imshow(window, frame)
                key = cv2.waitKey(0) & 0xFF
                if key in LABELS:
                    label = LABELS[key]
                    reviews[(dataset, episode.name)] = label
                    save_reviews(args.output, reviews)
                if key in (ord("q"), 27):
                    save_reviews(args.output, reviews)
                    print(f"Review paused; progress saved to {args.output}")
                    return
            if label:
                reviews[(dataset, episode.name)] = label
                save_reviews(args.output, reviews)
        save_reviews(args.output, reviews)
        print(f"Review complete; labels saved to {args.output}")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
