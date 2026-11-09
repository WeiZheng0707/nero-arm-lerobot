"""Show two OpenCV camera indices in one large window for role identification.

This is a camera-only diagnostic. It does not connect to or control a robot.
Press Q or Esc in either preview window to quit.
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indices", type=int, nargs=2, default=(1, 3))
    parser.add_argument("--backend", choices=("dshow", "msmf"), default="dshow")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--display-width", type=int, default=640)
    parser.add_argument("--display-height", type=int, default=480)
    args = parser.parse_args(argv)
    if (args.width <= 0 or args.height <= 0 or args.display_width <= 0
            or args.display_height <= 0 or any(index < 0 for index in args.indices)):
        parser.error("indices must be non-negative and dimensions must be positive")

    api = cv2.CAP_DSHOW if args.backend == "dshow" else cv2.CAP_MSMF
    opened: list[tuple[int, cv2.VideoCapture]] = []
    try:
        for index in args.indices:
            capture = cv2.VideoCapture(index, api)
            if not capture.isOpened():
                capture.release()
                raise RuntimeError(f"Could not open camera index={index} backend={args.backend}")
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            opened.append((index, capture))

        print("Camera preview only; no robot/CAN access. Press Q or Esc to quit.", flush=True)
        window = "NERO dual camera | base index 1 | wrist index 3"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, args.display_width * 2, args.display_height)

        while True:
            got_frame = False
            tiles = []
            for index, capture in opened:
                ok, frame = capture.read()
                if ok and frame is not None:
                    got_frame = True
                    role = "BASE" if index == 1 else "WRIST"
                    frame = cv2.resize(frame, (args.display_width, args.display_height),
                                       interpolation=cv2.INTER_LINEAR)
                    cv2.rectangle(frame, (0, 0), (args.display_width - 1, args.display_height - 1),
                                  (0, 220, 255), 5)
                    cv2.rectangle(frame, (16, 16), (330, 70), (0, 0, 0), -1)
                    cv2.putText(frame, f"{role} | index {index}", (30, 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
                    tiles.append(frame)
                else:
                    tiles.append(np.zeros(
                        (args.display_height, args.display_width, 3), dtype=np.uint8))
            if got_frame:
                canvas = cv2.hconcat(tiles)
                cv2.putText(canvas, "Q / Esc = QUIT", (30, canvas.shape[0] - 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow(window, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if not got_frame:
                raise RuntimeError("No frames received from either camera")
    finally:
        for _, capture in opened:
            capture.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
