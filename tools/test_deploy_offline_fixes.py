"""Offline checks for the deployment fixes. No camera, no CAN, no arm.

Covers four things that were wrong and are cheap to pin down without hardware:

1. limit_step must charge the MEASURED dt. Charging the nominal 1/fps while
   inference actually ran at ~10 Hz capped the real joint speed at a third of
   --max-joint-speed, which is why the arm crawled and then jumped every time a
   new action chunk started.
2. limit_step must also bound a single step absolutely, so a long stall (large
   dt) cannot turn into one large jump.
3. scene_status must not raise when a tracker has lost its target. A lost base14
   box is {'visible': 0} with no geometry keys, so the old code's
   scene['bag']['x'] raised KeyError exactly during occlusion.
4. plausible_cube_box must veto the cardboard-box false positive without
   touching a valid detection.

Run:
    python -X utf8 tools/test_deploy_offline_fixes.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from nero_dual_cam_act_deploy import (  # noqa: E402
    JOINTS,
    limit_step,
    plausible_cube_box,
    scene_status,
)

FAILURES = []


def check(name, condition, detail=""):
    if condition:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILURES.append(name)


def previous_pose():
    pose = {f"{name}.pos": 0.0 for name in JOINTS}
    pose["gripper.pos"] = 0.0
    return pose


def target_pose(delta):
    pose = {f"{name}.pos": delta for name in JOINTS}
    pose["gripper.pos"] = 1.0
    return pose


def test_measured_dt_is_honoured():
    print("limit_step: measured dt")
    # 0.3 rad/s for 100 ms allows 0.03 rad. The old nominal 1/30 s allowed only
    # 0.01 rad, i.e. a third of the requested speed.
    limited, clamped = limit_step(target_pose(0.05), previous_pose(), 0.3, 0.10)
    check("100ms dt permits 0.03 rad", abs(limited["joint_1.pos"] - 0.03) < 1e-9,
          f"got {limited['joint_1.pos']}")
    check("100ms dt reports clamped", clamped is True)

    # A small request well inside the bound must pass through untouched.
    limited, clamped = limit_step(target_pose(0.005), previous_pose(), 0.3, 0.10)
    check("small step passes through", abs(limited["joint_1.pos"] - 0.005) < 1e-9,
          f"got {limited['joint_1.pos']}")
    check("small step not clamped", clamped is False)

    # The old bug in one line: same request, nominal dt, one third the motion.
    nominal, _ = limit_step(target_pose(0.05), previous_pose(), 0.3, 1.0 / 30.0)
    check("nominal 1/30 dt would have allowed only 0.01 rad",
          abs(nominal["joint_1.pos"] - 0.01) < 1e-9, f"got {nominal['joint_1.pos']}")


def test_absolute_step_ceiling():
    print("limit_step: absolute per-step ceiling")
    # A 1 s dt would allow 0.3 rad at 0.3 rad/s; the ceiling must cap it.
    limited, clamped = limit_step(target_pose(0.5), previous_pose(), 0.3, 1.0,
                                  max_step_rad=0.05)
    check("ceiling caps a large dt", abs(limited["joint_1.pos"] - 0.05) < 1e-9,
          f"got {limited['joint_1.pos']}")
    check("ceiling reports clamped", clamped is True)

    # Without the ceiling the same input is much larger, proving the cap acts.
    uncapped, _ = limit_step(target_pose(0.5), previous_pose(), 0.3, 1.0)
    check("uncapped step is larger", uncapped["joint_1.pos"] > 0.05,
          f"got {uncapped['joint_1.pos']}")

    # The ceiling must not interfere when it is looser than the speed bound.
    limited, _ = limit_step(target_pose(0.05), previous_pose(), 0.3, 0.10,
                            max_step_rad=0.5)
    check("loose ceiling does not clamp", abs(limited["joint_1.pos"] - 0.03) < 1e-9,
          f"got {limited['joint_1.pos']}")


def test_gripper_not_limited():
    print("limit_step: gripper passthrough")
    limited, _ = limit_step(target_pose(0.0), previous_pose(), 0.001, 0.001)
    check("gripper keeps its value", limited["gripper.pos"] == 1.0,
          f"got {limited['gripper.pos']}")


def test_scene_status_survives_loss():
    print("scene_status: occluded targets")
    visible = {"cube": {"x": 0.4, "y": 0.6, "w": 0.05, "h": 0.07, "visible": 1},
               "bag": {"x": 0.2, "y": 0.2, "w": 0.2, "h": 0.3, "visible": 1}}
    text = scene_status(visible)
    check("visible scene reports coordinates", "cube=(0.40,0.60)" in text
          and "bag=(0.20,0.20)" in text, text)

    # This is the exact shape StableBox returns after it gives up: no geometry.
    lost = {"cube": {"visible": 0}, "bag": {"visible": 0}}
    try:
        text = scene_status(lost)
        ok = "cube=LOST" in text and "bag=LOST" in text
        check("lost targets do not raise and report LOST", ok, text)
    except KeyError as exc:
        check("lost targets do not raise and report LOST", False, f"KeyError: {exc}")

    # Half-lost: cube visible, bag gone. The old code indexed scene['bag']['x'].
    half = {"cube": visible["cube"], "bag": {"visible": 0}}
    try:
        text = scene_status(half)
        check("half-lost scene does not raise", "bag=LOST" in text, text)
    except KeyError as exc:
        check("half-lost scene does not raise", False, f"KeyError: {exc}")

    # A base14 scene has no ball key at all.
    check("no ball key is fine", "ball_visible" not in scene_status(lost))

    text = scene_status(lost, cube_vetoed=True)
    check("veto flag is shown", "vetoed" in text, text)


def test_cube_band_veto():
    print("plausible_cube_box: cardboard-box false positive")
    real = {"x": 0.39, "y": 0.59, "w": 0.05, "h": 0.07, "cx": 0.42, "cy": 0.62,
            "visible": 1}
    box, vetoed = plausible_cube_box(real, 0.35, 0.95)
    check("real cube is kept", box is real and vetoed is False)

    # Observed at outputs/deploy_b_30k_40s tick 399: the cube "jumped" to y=0.08,
    # which is the cardboard box at the top of the frame.
    false_positive = {"x": 0.32, "y": 0.02, "w": 0.09, "h": 0.12, "cx": 0.37,
                      "cy": 0.08, "visible": 1}
    box, vetoed = plausible_cube_box(false_positive, 0.35, 0.95)
    check("top-of-frame detection is vetoed", vetoed is True and box == {"visible": 0},
          str(box))

    already_lost = {"visible": 0}
    box, vetoed = plausible_cube_box(already_lost, 0.35, 0.95)
    check("already-lost box stays lost without veto flag",
          box is already_lost and vetoed is False)


def test_env_vector_after_veto():
    """The vetoed scene must still produce a valid 14-value base14 vector."""
    print("environment_state_base14: vetoed scene")
    from lerobot.nero_tools.env_state_schema import environment_state_base14

    scene = {"cube": {"visible": 0},
             "bag": {"x": 0.2, "y": 0.2, "w": 0.2, "h": 0.3, "cx": 0.3, "cy": 0.35,
                     "visible": 1, "area_ratio": 0.06}}
    values = environment_state_base14(scene)
    check("vector has 14 values", values.shape == (14,), str(values.shape))
    check("cube fields are zeroed", all(values[i] == 0 for i in range(5)), str(values))
    check("bag fields are kept", values[5] == 0.3 and values[9] == 1.0, str(values))
    check("relative offsets zeroed when cube lost",
          values[10] == 0 and values[11] == 0, str(values))


def main():
    test_measured_dt_is_honoured()
    test_absolute_step_ceiling()
    test_gripper_not_limited()
    test_scene_status_survives_loss()
    test_cube_band_veto()
    test_env_vector_after_veto()
    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): {', '.join(FAILURES)}")
        return 1
    print("ALL OFFLINE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
