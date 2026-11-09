# NERO hardware testing

- Before motion/enable tests, confirm physical support, empty grippers and a clear work area.
- User requirement (2026-09-21): any end effector enabled during a test must also be explicitly disabled at exit, including failure/interrupt cleanup. Seven joint enable bits do not cover grippers or teaching effectors.
- Verify disable using advancing feedback timestamps for joints AND each affected end effector. Report missing/stale feedback as UNCONFIRMED, never success. A separate pendant outside AgxGripper feedback needs separate verification.
- Read-only diagnostics must not enable, disable or change robot modes.
- Last verified mapping: channel 0 = leader firmware 1.20 (v120); channel 1 = follower firmware 1.21 (v121). This is NOT a permanent identity. Re-identify after USB changes, and never run legacy hardcoded-channel scripts blindly.
- Both end effectors were confirmed enabled while joint feedback was disabled. The cause was not established. Explicit disable_gripper requests then produced updated disabled feedback on both channels.
