"""Inspect identified follower; optionally disable only its gripper."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lerobot.nero_tools.test_enable_cycle import snapshot
from pyAgxArm import AgxArmFactory, create_agx_arm_config

parser = argparse.ArgumentParser()
parser.add_argument('--disable-gripper', action='store_true')
parser.add_argument('--can-control', action='store_true')
args = parser.parse_args()
arm = AgxArmFactory.create_arm(create_agx_arm_config(
    robot='nero', comm='can', interface='agx_cando', channel='0',
    bitrate=1000000, firmeware_version='v121'))
try:
    arm.connect()
    firmware = arm.get_firmware(timeout=3.0)
    print('FIRMWARE', firmware, flush=True)
    if not firmware or firmware.get('software_version') != '1.21':
        raise RuntimeError('Identity UNCONFIRMED; no disable sent')
    gripper = arm.init_effector(arm.OPTIONS.EFFECTOR.AGX_GRIPPER)
    time.sleep(1.0)
    previous = None
    advances = 0
    for sample in range(8):
        joints = snapshot(arm)
        status = arm.get_arm_status()
        if sample == 0 and status is not None:
            print('ARM STATUS', status, flush=True)
        grip = gripper.get_gripper_status()
        if grip is None or joints is None:
            raise RuntimeError('Feedback UNCONFIRMED')
        current = (list(joints[0]), float(grip.timestamp))
        fresh = previous is not None and all(a > b for a, b in zip(current[0], previous[0])) and current[1] > previous[1]
        enabled = bool(grip.msg.foc_status.driver_enable_status)
        print(json.dumps(dict(sample=sample, joint_stamps=current[0], joints_enabled=joints[1], joint_faults=joints[2],
            gripper_stamp=current[1], gripper_enabled=enabled, gripper_hz=grip.hz,
            mode=grip.msg.mode, value=grip.msg.value, fresh=fresh)), flush=True)
        if sample == 0 and args.disable_gripper:
            if args.can_control:
                if any(joints[1]):
                    raise RuntimeError('Joints enabled; refusing control-mode change')
                arm.set_motion_mode('j')
                time.sleep(0.5)
                mode = arm.get_arm_status()
                print('CONTROL AFTER SWITCH', mode, flush=True)
                if mode is None or int(mode.msg.ctrl_mode) != 1:
                    raise RuntimeError('CAN control UNCONFIRMED')
            for _ in range(3):
                gripper.disable_gripper()
                time.sleep(0.2)
            print('GRIPPER DISABLE SENT; no joint command', flush=True)
        if fresh and not enabled and not any(joints[1]):
            advances += 1
        else:
            advances = 0
        previous = current
        time.sleep(0.5)
    if args.disable_gripper and advances < 3:
        raise RuntimeError('Disabled state UNCONFIRMED')
    print('CONFIRMED joints and gripper disabled with advancing feedback' if advances >= 3 else 'INSPECTION COMPLETE', flush=True)
finally:
    arm.disconnect(join_timeout=1.0)
    print('Connection released', flush=True)
