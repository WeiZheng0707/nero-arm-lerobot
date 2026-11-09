"""Live eye-in-hand preview for colored object shape and gripper detection."""
from __future__ import annotations
import argparse
import time
import cv2
from lerobot.nero_tools.color_detector import detect_color_objects, detect_gripper


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--camera-backend", choices=("dshow", "msmf"), default="dshow")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--roi", default=None, help="x,y,width,height; omit to select tabletop ROI")
    args = parser.parse_args()
    candidates = list(dict.fromkeys([args.camera_index, 1, 0, 2, 3, 4]))

    def open_any_camera():
        backends = [args.camera_backend, "msmf" if args.camera_backend == "dshow" else "dshow"]
        for backend in backends:
            api = cv2.CAP_DSHOW if backend == "dshow" else cv2.CAP_MSMF
            for candidate in candidates:
                capture = cv2.VideoCapture(candidate, api)
                if not capture.isOpened():
                    capture.release()
                    continue
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                capture.set(cv2.CAP_PROP_FPS, args.fps)
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                for _ in range(8):
                    ok, frame = capture.read()
                    if ok and frame is not None:
                        print(f"Camera opened: requested={args.camera_index}, "
                              f"selected={candidate}, backend={backend}", flush=True)
                        return capture
                    time.sleep(0.05)
                capture.release()
        raise RuntimeError(f"No usable camera; requested={args.camera_index}, tried={candidates}")

    camera = open_any_camera()
    window = "NERO eye-in-hand detector preview"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 960, 720)
    colors = ("red", "green", "blue", "yellow")
    if args.roi is None:
        ok, first = camera.read()
        if not ok or first is None:
            raise RuntimeError("Camera opened but produced no initial frame")
        roi = tuple(map(int, cv2.selectROI(window, first, showCrosshair=True)))
        if roi[2] <= 0 or roi[3] <= 0:
            raise RuntimeError("Select a tabletop ROI before continuing")
    else:
        roi = tuple(map(int, args.roi.split(",")))
    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                camera.release()
                print("Camera read failed; reconnecting...", flush=True)
                camera = open_any_camera()
                ok, frame = camera.read()
                if not ok or frame is None:
                    raise RuntimeError("Camera frame read failed after reconnect")
            view = frame.copy()
            row = 28
            for color in colors:
                objects = detect_color_objects(frame, color, roi=roi)
                color_bgr = {"red": (0, 0, 255), "green": (0, 255, 0), "blue": (255, 0, 0), "yellow": (0, 220, 255)}[color]
                for index, obj in enumerate(objects):
                    x, y, w, h = (obj[key] for key in ("x", "y", "w", "h"))
                    cv2.rectangle(view, (x, y), (x + w, y + h), color_bgr, 2)
                    cv2.putText(view, f"{color} {obj['shape']} {index + 1}", (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)
                cv2.putText(view, f"{color}: {len(objects)}", (10, row), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
                row += 24
            gripper = detect_gripper(frame)
            if gripper:
                x, y, w, h = (gripper[key] for key in ("x", "y", "w", "h"))
                cv2.rectangle(view, (x, y), (x + w, y + h), (255, 0, 255), 2)
                cv2.putText(view, "gripper", (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 0, 255), 2)
            cv2.putText(view, "S=save  Q/Esc=quit", (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            cv2.rectangle(view, (roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3]), (180, 180, 180), 1)
            cv2.imshow(window, view)
            key = cv2.waitKey(max(1, int(1000 / args.fps))) & 0xFF
            if key == ord("r"):
                selected = tuple(map(int, cv2.selectROI(window, frame, showCrosshair=True)))
                if selected[2] > 0 and selected[3] > 0:
                    roi = selected
            if key == ord("s"):
                path = f"outputs/eye_in_hand_detector_{int(time.time())}.png"
                cv2.imwrite(path, view)
                print(f"Saved {path}", flush=True)
            if key in (ord("q"), 27):
                return 0
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
