"""Convert paired capture rows to local LeRobot datasets without robot access."""

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.nero_tools.env_state_schema import (
    ENVIRONMENT_STATE_BASE14_NAMES, validate_base14,
    ENVIRONMENT_STATE_RELATION_V2_NAMES,
    environment_state_relation_v2,
)


NAMES = [f"joint_{i}" for i in range(1, 8)] + ["gripper"]
DEFAULT_TASK = "Pick up the red cube and put it into the bag."
ENVIRONMENT_STATE = "observation.environment_state"

# 14-dim layout: cube (5) + bag (4) + ball (5). The cube is the target and the
# bag may move between episodes; the ball carries its own position because the
# B-near group exists specifically to teach selection ("which one is the
# target?"), and a bare presence bit would not say where the distractor is.
# Ball fields are zero/false in the 220 cube-only episodes, so one policy serves
# both collection groups.
ENVIRONMENT_STATE_NAMES = [
    "cube_x", "cube_y", "cube_w", "cube_h", "cube_visible",
    "bag_x", "bag_y", "bag_w", "bag_h",
    "ball_x", "ball_y", "ball_w", "ball_h", "ball_visible",
]
LEGACY_5D_NAMES = ["cube_x", "cube_y", "cube_w", "cube_h", "cube_visible"]
PI05_STATE_NAMES = [n + ".pos" for n in NAMES] + ENVIRONMENT_STATE_NAMES


def pi05_state_vector(robot_state, env_state):
    robot = np.asarray(robot_state, dtype=np.float32)
    env = np.asarray(env_state, dtype=np.float32)
    if robot.shape != (8,) or env.shape != (14,):
        raise ValueError("pi05_22 requires robot state (8,) and environment state (14,)")
    result = np.concatenate((robot, env))
    if not np.isfinite(result).all():
        raise ValueError("pi05_22 state contains non-finite values")
    return result


def detect_label_schema(frames):
    """Return 'base' for dual-camera label rows, 'legacy5' for old wrist rows, else None."""
    keys = set()
    for frame in frames:
        if "cube" in frame:
            keys.add("base")
        elif "block_visible" in frame:
            keys.add("legacy5")
        else:
            keys.add(None)
    if None in keys:
        return None
    if len(keys) > 1:
        raise ValueError("Episode mixes base-camera and legacy wrist labels")
    return keys.pop()


def environment_state_vector(frame, schema):
    """Build the fixed-order environment_state row for one frame.

    The dual-camera schema stores each object as {'x','y','w','h','visible'};
    a missing object becomes zeros so a dropped detection never shifts columns.
    Column order is exactly ENVIRONMENT_STATE_NAMES.
    """
    if schema == 'base14':
        values = frame.get('base14') or {}
        if set(values) != set(ENVIRONMENT_STATE_BASE14_NAMES):
            raise ValueError('base14 field names do not match schema')
        return validate_base14([values[name] for name in ENVIRONMENT_STATE_BASE14_NAMES])
    if schema == "legacy5":
        return np.asarray([frame.get(name, 0.0) for name in LEGACY_5D_NAMES], dtype=np.float32)
    if schema == "relation_v2":
        values = frame.get("relation_v2") or {}
        missing = [name for name in ENVIRONMENT_STATE_RELATION_V2_NAMES if name not in values]
        if missing:
            raise ValueError(f"relation_v2 row is missing fields: {missing}")
        result = np.asarray([values[name] for name in ENVIRONMENT_STATE_RELATION_V2_NAMES], dtype=np.float32)
        if not np.isfinite(result).all():
            raise ValueError("relation_v2 environment state contains non-finite values")
        return result
    values = []
    for field in ("cube", "bag", "ball"):
        box = frame.get(field) or {}
        for key in ("x", "y", "w", "h"):
            values.append(float(box.get(key, 0.0)))
        # cube carries its visibility at idx 4; bag at idx 9 is implied by w/h;
        # ball carries visibility last so the layout matches the name list.
        if field == "cube":
            values.append(float(box.get("visible", 0.0)))
    ball = frame.get("ball") or {}
    values.append(float(ball.get("visible", 0.0)))
    return np.asarray(values, dtype=np.float32)


def configure_camera_features(features, camera_map, use_video, crop_by_camera=None,
                              scale_by_camera=None):
    """Add mapped camera features using the requested LeRobot storage dtype.

    `crop_by_camera` maps a camera name to (x, y, w, h); a cropped camera is
    declared at the cropped size so the feature shape matches the written frames.
    `scale_by_camera` then resizes those pixels by a factor, and the declared
    shape follows that too, otherwise the reload assertion fails.
    """
    crop_by_camera = crop_by_camera or {}
    scale_by_camera = scale_by_camera or {}
    features = dict(features)
    features.pop("observation.images.wrist", None)
    dtype = "video" if use_video else "image"
    for cam, feature in camera_map.items():
        if cam in crop_by_camera:
            _, _, crop_w, crop_h = crop_by_camera[cam]
        else:
            crop_w, crop_h = 640, 480
        factor = scale_by_camera.get(cam, 1.0)
        width = max(1, int(round(crop_w * factor)))
        height = max(1, int(round(crop_h * factor)))
        features[feature] = {
            "dtype": dtype,
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        }
    return features


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def merge_env_state(path, frames, env_state_file="env_state.jsonl", env_state_schema="legacy14"): 
    """Attach offline-computed cube/ball/bag labels to the frame rows.

    Capture writes only robot state and images and records
    `env_state_status: deferred_raw_capture`, because running colour detection
    inside the 30 Hz control loop would steal time from teleoperation. The
    labels are produced afterwards by `tools/env_state_detect.py`, which writes
    `env_state.jsonl` next to `timestamps.jsonl`.

    The two files are matched by row index, so a length mismatch is a hard
    error: silently zipping a truncated label file would shift every position
    label and poison the dataset.
    """
    labels_path = path / env_state_file
    if not labels_path.exists():
        return frames, False
    labels = rows(labels_path)
    if len(labels) != len(frames):
        raise ValueError(
            f"{path}: env_state.jsonl has {len(labels)} rows but timestamps.jsonl "
            f"has {len(frames)}; re-run tools/env_state_detect.py for this episode")
    for index, (frame, label) in enumerate(zip(frames, labels, strict=True)):
        if env_state_schema == 'base14':
            if label.get('schema') != 'base14' or label.get('frame_index') != index:
                raise ValueError(f'{path}: base14 schema/frame mismatch at {index}')
            if label.get('host_time_s') != frame.get('host_time_s'):
                raise ValueError(f'{path}: base14 timestamp mismatch at {index}')
            environment_state_vector(label, 'base14')
            frame['base14'] = label['base14']
            frame['base14_config_hash'] = label.get('detector_config_hash')
            continue
        if env_state_schema == "relation_v2":
            if "relation_v2" not in label:
                raise ValueError(f"{path}: {env_state_file} row {index} has no relation_v2 key")
            frame["relation_v2"] = label.get("relation_v2") or {}
            frame["labels_source"] = "base_plus_wrist_relation_v2"
            continue
        if "cube" not in label:
            raise ValueError(f"{path}: env_state row {index} has no 'cube' key")
        frame["cube"] = label.get("cube") or {}
        frame["ball"] = label.get("ball") or {}
        frame["bag"] = label.get("bag") or {}
        frame["labels_source"] = "base_env_state_jsonl"
    return frames, True


def inspect_episode(path, expect_cameras=None, env_state_file="env_state.jsonl",
                    env_state_schema="legacy14"): 
    """Validate one raw episode.

    `expect_cameras` is None for the legacy single-image layout, or a tuple of
    camera keys (e.g. ('wrist', 'base')) for the paired dual-camera layout.
    """
    frames, states, actions = [rows(path / name) for name in
                               ("timestamps.jsonl", "robot_state.jsonl", "actions.jsonl")]
    if not frames or not len(frames) == len(states) == len(actions):
        raise ValueError(f"{path}: unequal or empty row counts")
    frames, _ = merge_env_state(path, frames, env_state_file, env_state_schema)
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if metadata["result"] not in ("auto_end", "manual_end"):
        raise ValueError(f"{path}: requires manual review of {metadata['result']}")
    state = np.asarray([[row.get(n, row.get(n + ".pos")) for n in NAMES] for row in states], dtype=np.float32)
    action = np.asarray([[row[n + ".pos"] for n in NAMES] for row in actions], dtype=np.float32)
    stamps = np.asarray([row["host_time_s"] for row in frames], dtype=np.float64)
    if not np.isfinite(state).all() or not np.isfinite(action).all() or not np.isfinite(stamps).all():
        raise ValueError(f"{path}: nonfinite values")
    if np.any(np.diff(stamps) <= 0):
        raise ValueError(f"{path}: non-increasing timestamps")
    for i, frame in enumerate(frames):
        if expect_cameras is None:
            image_keys = (frame["image"],)
        else:
            present = set(frame.get("cameras", {}))
            if present != set(expect_cameras):
                raise ValueError(f"{path}: expected cameras {expect_cameras}, got {sorted(present)}")
            image_keys = tuple(frame["cameras"][key]["image"] for key in expect_cameras)
        for relative in image_keys:
            image_path = (path / relative).resolve()
            if not image_path.is_relative_to(path.resolve()):
                raise ValueError("Image path escapes episode")
            with Image.open(image_path) as image:
                if image.size != (640, 480):
                    raise ValueError(f"{image_path}: unexpected resolution")
                image.verify()
        if max(abs(states[i]["host_time_s"] - stamps[i]),
               abs(actions[i]["host_time_s"] - stamps[i])) > 0.05:
            raise ValueError(f"{path}: row alignment exceeds 50 ms")
    if np.any(state[:, -1] < 0) or np.any(state[:, -1] > 1) or np.any(action[:, -1] < 0) or np.any(action[:, -1] > 1):
        raise ValueError(f"{path}: gripper outside [0,1]")
    return frames, state, action, stamps, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True,
                        help="Raw episode directory; repeat once per source dataset")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--dataset-name", default="nero_red_cube")
    parser.add_argument("--state-mode", choices=("act8", "pi05_22"), default="act8")
    parser.add_argument("--image-key", action="append", default=None, metavar="CAM=FEATURE",
                        help="Map a raw camera directory to a dataset image feature, "
                             "repeatable. Example: 'wrist=observation.images.wrist' "
                             "'base=observation.images.base'. Omit for the legacy "
                             "single-image layout.")
    parser.add_argument("--env-state-file", default="env_state.jsonl",
                        help="Label filename in each episode; relation_v2 is separate from legacy labels")
    parser.add_argument('--labels-root', type=Path, help='Separate label root containing episode directories; one source only')
    parser.add_argument("--env-state-schema", choices=("legacy14", "relation_v2", "base14"),
                        default="legacy14", help="Feature layout to read from --env-state-file")
    parser.add_argument("--no-images", action="store_true",
                        help="Write only state, action and env-state; no image features. "
                             "Used to train the env-state-only control model, which tests "
                             "whether the detected positions carry enough information and "
                             "cannot fall back on the cameras.")
    parser.add_argument("--crop", action="append", default=None, metavar="CAM=X,Y,W,H",
                        help="Crop one camera to this pixel rectangle before writing. "
                             "Repeatable. Cropping the static base view to the working "
                             "area removes most of the background tokens, which is the "
                             "difference between the cube occupying ~1 ResNet patch and "
                             "being findable at all.")
    parser.add_argument("--scale", action="append", default=None, metavar="CAM=FACTOR",
                        help="Resize this camera by FACTOR after any crop. Repeatable. "
                             "At 640x480 the cube spans about 25 px and ResNet-18's "
                             "32x downsample leaves it inside a single feature patch, "
                             "which limits how precisely the policy can steer by it; "
                             "2.0 gives it roughly four patches.")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Keep only the first N frames of each episode (30 Hz). The "
                             "captured 20 s episodes end with waiting and return-to-home; "
                             "keeping that tail teaches the policy to stand still.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--video", action="store_true",
                        help="Store the two camera streams as per-episode mp4 instead of "
                             "PNG frames embedded in parquet. Roughly 20-30x smaller on "
                             "disk; costs encode time and one extra PNG write pass.")
    parser.add_argument("--video-codec", default="h264", choices=("h264", "hevc", "libsvtav1"),
                        help="h264 is the safe default: it ships with the PyAV wheel and "
                             "decodes everywhere. LeRobot's own default is libsvtav1, which "
                             "needs an ffmpeg build with SVT-AV1 and encodes far slower.")
    parser.add_argument("--video-crf", type=int, default=20,
                        help="Constant-rate factor. Lower is better quality and bigger. "
                             "LeRobot defaults to 30, which is aggressive for h264 and can "
                             "smear a 30 px cube; 18-23 is the sensible range here.")
    parser.add_argument("--video-pix-fmt", default="yuv420p",
                        choices=("yuv420p", "yuv444p"),
                        help="yuv420p halves chroma resolution but is the only format every "
                             "decoder and browser plays; yuv444p keeps full colour for h264.")
    parser.add_argument("--video-backend", default=None, choices=("torchcodec", "pyav"),
                        help="Decoder used to verify the written videos. LeRobot defaults to "
                             "torchcodec whenever the package is merely installed, even if it "
                             "cannot load its shared libraries; pass 'pyav' here when "
                             "torchcodec import fails.")
    parser.add_argument("--review-csv", type=Path,
                        help="Human review CSV; only rows labeled success are included")
    parser.add_argument("--visual-reviewed", action="store_true",
                        help="Set only after a human has confirmed each episode actually "
                             "succeeded. The manifest records this so later sessions do not "
                             "mistake an unreviewed dataset for a verified one.")
    parser.add_argument("--operator-confirmed-all-success", action="store_true",
                        help="Operator states every episode is a correct demonstration, even "
                             "though metadata says auto_end (the run hit the time limit instead "
                             "of pressing E). Without this, auto_end is still converted but "
                             "the manifest records the labels as unverified. With it, the "
                             "manifest records the claim and its exact wording, so nobody "
                             "later mistakes an asserted label for a machine-verified one.")
    args = parser.parse_args()
    if args.labels_root is not None and len(args.source) != 1:
        parser.error('--labels-root requires exactly one source')
    if args.env_state_schema == 'base14':
        if not args.video or args.no_images:
            parser.error('base14 workflow requires --video and camera images')
        if args.env_state_file == 'env_state.jsonl':
            args.env_state_file = 'env_state_base14.jsonl'
    if args.visual_reviewed and args.review_csv is None:
        parser.error("--visual-reviewed requires --review-csv")
    if args.operator_confirmed_all_success and args.review_csv is not None:
        parser.error("--operator-confirmed-all-success cannot be combined with "
                     "--review-csv; the review CSV is the stronger, per-episode record")
    camera_map = {}
    if args.no_images:
        if args.image_key:
            parser.error("--no-images cannot be combined with --image-key")
    elif args.image_key:
        for value in args.image_key:
            cam, _, feature = value.partition("=")
            if not cam or not feature:
                parser.error(f"--image-key needs CAM=FEATURE: {value!r}")
            camera_map[cam.strip()] = feature.strip()
    expect_cameras = tuple(camera_map) if camera_map else None
    crop_by_camera = {}
    if args.crop:
        for value in args.crop:
            cam, _, rect = value.partition("=")
            if not cam or not rect:
                parser.error(f"--crop needs CAM=X,Y,W,H: {value!r}")
            parts = [int(part) for part in rect.replace(" ", "").split(",")]
            if len(parts) != 4 or parts[2] <= 0 or parts[3] <= 0:
                parser.error(f"--crop needs positive W,H: {value!r}")
            crop_by_camera[cam.strip()] = tuple(parts)
        unknown = set(crop_by_camera) - set(camera_map)
        if unknown:
            parser.error(f"--crop names a camera that is not in --image-key: {sorted(unknown)}")
    scale_by_camera = {}
    if args.scale:
        for value in args.scale:
            cam, _, factor = value.partition("=")
            try:
                factor_value = float(factor)
            except ValueError:
                parser.error(f"--scale needs CAM=FACTOR: {value!r}")
            if not cam or not 0.1 <= factor_value <= 4.0:
                parser.error(f"--scale needs a factor between 0.1 and 4.0: {value!r}")
            scale_by_camera[cam.strip()] = factor_value
        unknown = set(scale_by_camera) - set(camera_map)
        if unknown:
            parser.error(f"--scale names a camera that is not in --image-key: {sorted(unknown)}")
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"Refusing existing destination: {args.output}")
    sources = [(source.name, source, sorted(source.glob("episode_*"))) for source in args.source]
    if any(len(paths) < 2 for _, _, paths in sources):
        raise ValueError("Each source must contain at least two episodes")

    if args.review_csv is not None:
        with args.review_csv.open(newline="", encoding="utf-8-sig") as review_file:
            review_rows = list(csv.DictReader(review_file))
        expected = {
            (label, path.name)
            for label, _, paths in sources
            for path in paths
        }
        reviews = {}
        for row in review_rows:
            key = (row.get("dataset", ""), row.get("episode", ""))
            label = row.get("label", "").strip().lower()
            if key not in expected:
                raise ValueError(f"Unknown episode in review CSV: {key}")
            if key in reviews:
                raise ValueError(f"Duplicate review row: {key}")
            if label not in {"success", "failure", "uncertain"}:
                raise ValueError(f"Invalid review label for {key}: {label!r}")
            reviews[key] = label
        missing = expected - reviews.keys()
        if missing:
            raise ValueError(f"Review every source episode first; missing {len(missing)} labels")
        sources = [
            (label, source, [path for path in paths if reviews[(label, path.name)] == "success"])
            for label, source, paths in sources
        ]
        if any(len(paths) < 2 for _, _, paths in sources):
            raise ValueError("Each reviewed source needs at least two successful episodes")
        print("REVIEW FILTER: only explicitly successful episodes will be converted", flush=True)

    if args.operator_confirmed_all_success:
        print("SUCCESS LABELS: operator asserted every episode is correct "
              "(metadata auto_end). Recorded in the manifest as "
              "'operator_asserted_auto_end'.", flush=True)

    if args.video:
        # LeRobot's encode_episode_videos() calls encode_video_frames() with its own
        # defaults (libsvtav1, crf 30). libsvtav1 is not in every PyAV build and
        # encodes roughly an order of magnitude slower than h264, so the codec is
        # bound here instead of editing the upstream module.
        import functools

        import lerobot.datasets.lerobot_dataset as _dataset_module
        from lerobot.datasets.video_utils import encode_video_frames as _encode

        _dataset_module.encode_video_frames = functools.partial(
            _encode, vcodec=args.video_codec, crf=args.video_crf,
            pix_fmt=args.video_pix_fmt)
        print(f"VIDEO MODE: codec={args.video_codec} crf={args.video_crf} "
              f"pix_fmt={args.video_pix_fmt}. Frames are written as PNG, encoded, then "
              f"the images/ directory is deleted after each episode.", flush=True)

    records = [(label, source, path) for label, source, paths in sources for path in paths]
    audited = {}
    split_by_path = {}
    for label, source, paths in sources:
        shuffled = list(paths)
        random.Random(42).shuffle(shuffled)
        val_paths = {p.resolve() for p in shuffled[:max(1, round(len(paths) * 0.2))]}
        for path in paths:
            key = str(path.resolve())
            audited[key] = inspect_episode(path, expect_cameras=expect_cameras,
                                           env_state_file=(args.labels_root.resolve() / path.name / args.env_state_file
                                                           if args.labels_root else args.env_state_file),
                                           env_state_schema=args.env_state_schema)
            split_by_path[key] = "val" if path.resolve() in val_paths else "train"
            print(f"AUDITED {label}/{path.name}: {len(audited[key][0])} frames", flush=True)
    if args.env_state_schema == 'base14':
        schemas = {'base14' if item[0][0].get('base14') is not None else None
                   for item in audited.values()}
        if schemas != {'base14'}:
            raise ValueError('Every episode requires complete base14 labels before conversion')
        hashes = {frame.get('base14_config_hash') for item in audited.values() for frame in item[0]}
        if len(hashes) != 1 or not next(iter(hashes)):
            raise ValueError('base14 detector configurations differ or are missing')
    elif args.env_state_schema == "relation_v2":
        # audited values are (frames, state, action, stamps, metadata); the label
        # row lives in the first frame, so this needs [0][0]. Using [0] alone put a
        # list where a dict was expected and raised AttributeError before the
        # output directory was even created.
        schemas = {"relation_v2" if rows_for_episode[0][0].get("relation_v2") is not None else None
                   for rows_for_episode in audited.values()}
    else:
        schemas = {detect_label_schema(rows_for_episode[0])
                   for rows_for_episode in audited.values()}
    if len(schemas) > 1:
        missing = [key for key, value in audited.items()
                   if ((value[0][0].get("relation_v2") is None)
                       if args.env_state_schema == "relation_v2"
                       else detect_label_schema(value[0]) is None)]
        raise ValueError(
            "Mixed label schemas across episodes: " + repr(schemas) + ". "
            f"{len(missing)} episode(s) have no position labels at all. "
            f"Run tools/env_state_batch.py --schema {args.env_state_schema} for every episode (raw capture defers "
            "env-state), or convert only the episodes that already have labels. "
            "Refusing to write a dataset where some rows silently have no "
            "environment_state. First offenders: " + ", ".join(missing[:5]))
    label_schema = schemas.pop()
    if args.state_mode == "pi05_22" and label_schema != "base":
        raise ValueError("pi05_22 requires complete legacy14 base-camera labels in the documented order")
    if args.env_state_schema == "relation_v2" and label_schema is None:
        raise ValueError(
            "--env-state-schema relation_v2 was requested, but no audited episode "
            f"contains {args.env_state_file} relation_v2 labels. Generate v2 labels "
            "before conversion; refusing to create an image-only dataset by accident.")
    if label_schema is None:
        print("  [WARN] No episode has position labels, so no "
              "observation.environment_state will be written. That is correct for a "
              "wrist-only baseline, but the moving-bag design needs env-state: run "
              "tools/env_state_detect.py first if this was not intended.", flush=True)
    has_cube_annotations = label_schema is not None
    state_dim = (len(ENVIRONMENT_STATE_RELATION_V2_NAMES) if label_schema == "relation_v2" else
                 len(ENVIRONMENT_STATE_NAMES) if label_schema == "base" else len(LEGACY_5D_NAMES))
    state_names = (ENVIRONMENT_STATE_RELATION_V2_NAMES if label_schema == "relation_v2" else
                   ENVIRONMENT_STATE_NAMES if label_schema == "base" else LEGACY_5D_NAMES)
    if label_schema == 'base14':
        state_names = ENVIRONMENT_STATE_BASE14_NAMES
        state_dim = len(state_names)
    if label_schema == "legacy5":
        print("  [WARN] legacy 5-dim wrist labels detected; positions are NOT absolute "
              "coordinates. Re-capture with a base camera for the moving-bag design.",
              flush=True)
    features = {
        "observation.state": {"dtype": "float32", "shape": (8,), "names": [n + ".pos" for n in NAMES]},
        "action": {"dtype": "float32", "shape": (8,), "names": [n + ".pos" for n in NAMES]},
    }
    if not args.no_images:
        features["observation.images.wrist"] = {
            "dtype": "video" if args.video else "image",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
        }
    if camera_map:
        # Every mapped camera becomes an image feature. Capture stores both wrist
        # and base at 640x480; a `--crop` on a camera shrinks the stored feature,
        # so the declared shape must follow the crop, not the raw frame.
        features = configure_camera_features(features, camera_map, args.video,
                                             crop_by_camera, scale_by_camera)
        if crop_by_camera:
            print(f"IMAGE CROP: {crop_by_camera}", flush=True)
        if scale_by_camera:
            print(f"IMAGE SCALE: {scale_by_camera}", flush=True)
    if args.state_mode == "pi05_22":
        features["observation.state"] = {"dtype": "float32", "shape": (22,),
                                         "names": PI05_STATE_NAMES}
    elif has_cube_annotations:
        features[ENVIRONMENT_STATE] = {
            "dtype": "float32", "shape": (state_dim,), "names": state_names,
        }
    args.output.mkdir(parents=True, exist_ok=args.resume)
    manifest = {"sources": {label: str(source.resolve()) for label, source, _ in sources},
                "seed": 42, "fps": 30, "state_mode": args.state_mode,
                "task": args.task, "joint_unit": "rad", "gripper_unit": "normalized_0_1",
                "image_keys": camera_map,
                "storage": ("video" if args.video else "images_in_parquet"),
                "video": ({"codec": args.video_codec, "crf": args.video_crf,
                           "pix_fmt": args.video_pix_fmt, "fps": 30,
                           "decode_backend": args.video_backend or "auto"}
                          if args.video else None),
                "label_schema": label_schema,
                "base14_config_hash": next(iter(hashes)) if label_schema == 'base14' else None,
                "environment_state_names": state_names,
                "timing": "nearest paired capture row resampled on 30 Hz host-time grid; original times in alignment.jsonl",
                "visual_success_reviewed": bool(args.visual_reviewed),
                "success_label_source": (
                    "review_csv" if args.review_csv is not None
                    else "operator_asserted_auto_end" if args.operator_confirmed_all_success
                    else "unverified_auto_end"),
                "success_label_note": (
                    "Operator asserted every episode is a correct demonstration; metadata "
                    "says auto_end because the run ended on the time limit rather than the E "
                    "key. This is an assertion, not a machine verification."
                    if args.operator_confirmed_all_success else
                    "Only episodes passing --review-csv are verified successes."
                    if args.review_csv is not None else
                    "No human review supplied: auto_end does NOT prove success."),
                "splits": {}}
    for split in ("train", "val"):
        selected = [(label, source, path) for label, source, path in records
                    if split_by_path[str(path.resolve())] == split]
        root = args.output / split
        repo_id = f"local/{args.dataset_name}_{split}"
        if root.exists() and args.resume:
            loaded = LeRobotDataset(repo_id, root=root, video_backend=args.video_backend)
            if loaded.meta.features["observation.state"]["shape"] != list(features["observation.state"]["shape"]):
                raise ValueError("Resume state shape differs from requested state mode")
            align_rows = rows(root / "alignment.jsonl")
            assert loaded.num_episodes == len(selected)
            assert len(loaded) == len(align_rows)
            entries = []
            for index, (label, _, path) in enumerate(selected):
                mapped = [r for r in align_rows if r["episode_index"] == index]
                assert mapped and all(r["source_episode"] == path.name and
                                      r["source_dataset"] == label for r in mapped)
                sample = loaded[int(loaded.episode_data_index["from"][index])]
                if camera_map:
                    for cam, feature in camera_map.items():
                        declared = features[feature]["shape"]
                        assert tuple(sample[feature].shape) == (3, declared[0], declared[1]), (
                            f"{cam}: {tuple(sample[feature].shape)} != "
                            f"(3, {declared[0]}, {declared[1]})")
                else:
                    if not args.no_images:
                        assert tuple(sample["observation.images.wrist"].shape) == (3, 480, 640)
                entries.append({"episode_index": index, "source_dataset": label,
                                "source_episode": path.name,
                                "source_frames": len(audited[str(path.resolve())][0]),
                                "frames": len(mapped),
                                "source_metadata": audited[str(path.resolve())][4]})
            manifest["splits"][split] = {"repo_id": repo_id, "root": str(root.resolve()), "episodes": entries}
            print(f"RESUMED completed {split}: {len(selected)} episodes", flush=True)
            continue
        ds = LeRobotDataset.create(repo_id, fps=30, root=root, robot_type="nero_follower",
                                   features=features, use_videos=args.video,
                                   video_backend=args.video_backend)
        entries = []
        with (root / "alignment.jsonl").open("w", encoding="utf-8") as align:
            for index, (label, _, path) in enumerate(selected):
                frames, state, action, stamps, metadata = audited[str(path.resolve())]
                relative = stamps - stamps[0]
                grid = np.arange(int(np.floor(relative[-1] * 30)) + 1) / 30
                right = np.clip(np.searchsorted(relative, grid), 0, len(relative)-1)
                left = np.maximum(right-1, 0)
                indices = np.where(abs(relative[left]-grid) <= abs(relative[right]-grid), left, right)
                if args.max_frames is not None:
                    # The captured episodes run to 20 s, but the useful part of the
                    # demonstration is the first ~13 s; the tail is waiting and
                    # return-to-home. Keeping it teaches the policy to stand still.
                    indices = indices[:args.max_frames]
                    # The grid must be trimmed alongside the indices: the
                    # resampling-error check below compares them element-wise, and
                    # a full-length grid raised "operands could not be broadcast
                    # together with shapes (480,) (600,)".
                    grid = grid[:len(indices)]
                for frame_index, src in enumerate(indices):
                    frame_data = {"observation.state": state[src], "action": action[src]}
                    if camera_map:
                        for cam, feature in camera_map.items():
                            relative_image = (frames[src]["cameras"][cam]["image"]
                                              if expect_cameras else frames[src]["image"])
                            with Image.open(path / relative_image) as image:
                                rgb = np.asarray(image.convert("RGB")).copy()
                            if cam in crop_by_camera:
                                x, y, w, h = crop_by_camera[cam]
                                if y + h > rgb.shape[0] or x + w > rgb.shape[1]:
                                    raise ValueError(
                                        f"{path}: {cam} crop {crop_by_camera[cam]} exceeds "
                                        f"image {rgb.shape[1]}x{rgb.shape[0]}")
                                rgb = rgb[y:y + h, x:x + w]
                            if cam in scale_by_camera:
                                import cv2 as _cv2
                                factor = scale_by_camera[cam]
                                rgb = _cv2.resize(
                                    rgb, None, fx=factor, fy=factor,
                                    interpolation=_cv2.INTER_CUBIC)
                            frame_data[feature] = rgb
                    else:
                        if not args.no_images:
                            with Image.open(path / frames[src]["image"]) as image:
                                frame_data["observation.images.wrist"] = np.asarray(
                                    image.convert("RGB")).copy()
                    if args.state_mode == "pi05_22":
                        frame_data["observation.state"] = pi05_state_vector(
                            state[src], environment_state_vector(frames[src], label_schema))
                    elif has_cube_annotations:
                        frame_data[ENVIRONMENT_STATE] = environment_state_vector(
                            frames[src], label_schema)
                    ds.add_frame(frame_data, task=args.task)
                    align_row = {"episode_index": index, "frame_index": frame_index,
                                 "source_dataset": label,
                                 "source_episode": path.name, "source_row": int(src),
                                 "source_host_time_s": float(stamps[src])}
                    if expect_cameras:
                        align_row["camera_timestamp_ms"] = {
                            cam: frames[src]["cameras"][cam].get("camera_timestamp_ms")
                            for cam in expect_cameras}
                    else:
                        align_row["camera_timestamp_ms"] = frames[src].get("camera_timestamp_ms")
                    if has_cube_annotations:
                        align_row["environment_state"] = environment_state_vector(
                            frames[src], label_schema).tolist()
                    align.write(json.dumps(align_row) + "\n")
                ds.save_episode()
                entries.append({"episode_index": index, "source_dataset": label,
                                "source_episode": path.name,
                                "source_frames": len(frames), "frames": len(grid),
                                "max_source_gap_s": float(np.diff(stamps).max()),
                                "max_resampling_error_s": float(abs(relative[indices]-grid).max()),
                                "source_metadata": metadata})
                print(f"SAVED {split}/{index}: {path.name}, {len(grid)} frames", flush=True)
        loaded = LeRobotDataset(repo_id, root=root, video_backend=args.video_backend)
        assert loaded.num_episodes == len(selected)
        assert len(loaded) == sum(e["frames"] for e in entries)
        for bounds in zip(loaded.episode_data_index["from"], loaded.episode_data_index["to"]):
            for sample_index in (int(bounds[0]), int(bounds[1])-1):
                sample = loaded[sample_index]
                if camera_map:
                    for cam, feature in camera_map.items():
                        declared = features[feature]["shape"]
                        assert tuple(sample[feature].shape) == (3, declared[0], declared[1]), (
                            f"{cam} image shape {tuple(sample[feature].shape)} != "
                            f"(3, {declared[0]}, {declared[1]})")
                else:
                    if not args.no_images:
                        assert tuple(sample["observation.images.wrist"].shape) == (3, 480, 640)
                assert tuple(sample["observation.state"].shape) == features["observation.state"]["shape"]
                if has_cube_annotations and args.state_mode == "act8":
                    assert tuple(sample[ENVIRONMENT_STATE].shape) == (state_dim,)
        if label_schema == 'base14':
            import av
            for episode_index, entry in enumerate(entries):
                for camera_key in loaded.meta.video_keys:
                    video_path = root / loaded.meta.get_video_file_path(episode_index, camera_key)
                    with av.open(str(video_path)) as container:
                        decoded_count = sum(1 for _ in container.decode(video=0))
                    if decoded_count != entry['frames']:
                        raise ValueError(f'{video_path}: decoded {decoded_count}, expected {entry["frames"]}')
        manifest["splits"][split] = {"repo_id": repo_id, "root": str(root.resolve()), "episodes": entries}
    (args.output / "conversion_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if args.video:
        expected_videos = len(records) * (len(camera_map) if camera_map else 1)
        actual_videos = len(list(args.output.rglob("*.mp4")))
        if actual_videos != expected_videos:
            raise RuntimeError(
                f"Video conversion incomplete: expected {expected_videos} MP4 files, "
                f"found {actual_videos}")
        print(f"VIDEO CHECK: {actual_videos}/{expected_videos} MP4 files", flush=True)
    print("DONE: both datasets reloaded and episode boundary samples verified.", flush=True)


if __name__ == "__main__":
    main()
