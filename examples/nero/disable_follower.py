"""Request disable on the supported NERO follower. Requires physical support first."""

import sys
import time
from pathlib import Path


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src/lerobot/motors/nero"))
    from read_nero_firmware import preflight_can_device
    from pyAgxArm import AgxArmFactory, create_agx_arm_config

    if not preflight_can_device("agx_cando", "0"):
        raise RuntimeError("Cannot open follower channel 0; disable not sent")
    config = create_agx_arm_config(
        robot="nero", comm="can", interface="agx_cando", channel="0",
        bitrate=1_000_000, firmeware_version="default",
    )
    arm = AgxArmFactory.create_arm(config)
    try:
        arm.connect()
        for attempt in range(3):
            arm.disable(255)
            print(f"Follower channel 0: disable request {attempt + 1}/3 sent", flush=True)
            time.sleep(0.3)
        print("DISABLE REQUESTS SENT. Physical disable remains UNCONFIRMED; feedback was unreliable.",
              flush=True)
    finally:
        arm.disconnect(join_timeout=1.0)
        print("Connection cleanup completed.", flush=True)


if __name__ == "__main__":
    main()
