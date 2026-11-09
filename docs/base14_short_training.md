# Base14 short training handoff — 2026-11-02

Verified dataset: D:/NERO/outputs/red_cube_base14_video_v1.
Train: 240 episodes / 143932 frames / 480 H.264 videos.
Validation: 60 episodes / 35979 frames / 120 H.264 videos.
Total disk usage: 2.7344 GiB. Split source episodes are disjoint.
All Parquet indices and environment vectors match alignment.jsonl; state,
action and environment values are finite. All MP4 header frame counts and
codecs match the manifest. Conversion itself also performs full video decoding.
The training smoke check decoded actual samples and ran forward/backward on
both models using RTX 4090. This is a two-sample smoke check, not a completed
training run or proof of grasp success. Report: outputs/base14_training_ready.json.

In Anaconda Prompt activate lerobot, then run from this project:

    train_base14_short.cmd a
    train_base14_short.cmd b

Run sequentially. A uses wrist image and robot state. B additionally uses base14.
Both use seed 42, batch 8, chunk 60, 2000 steps, no image/environment dropout,
and explicit MEAN_STD normalization. Both retain the same full trajectories.
Windows data loader uses zero workers initially. Outputs are separate new
D:/NERO/outputs/train/base14_a_2k_v1 and base14_b_2k_v1 directories.
Dataset cache is D:/NERO/cache/datasets. Existing training directories are rejected.

After both finish, evaluate their final pretrained_model directories with
tools/evaluate_base14_policy.py against the same validation root and repo ID
local/nero_red_cube_base14_v1_val. Compare future joint and gripper errors,
hold-state baseline, and B's environment interventions. Do not approve robot
execution from training loss or environment sensitivity alone. Short training
is an initial checkpoint; a weak result at 2000 steps is not final evidence
that the environment signal is useless.
