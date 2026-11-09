# PI0.5 LoRA for the current red-cube dataset

Status: preparation only. Environment installation was not approved. User
provided outputs/model: config parses as PI05Config (gemma_2b/gemma_300m),
model.safetensors has 812 F32 tensors, 14467165872 bytes; header offsets match
file length. This is structural validation, not a source checksum or model
forward test. Tokenizer files are absent from that directory. No forward/backward or deployment
test has been completed. Do not use the old train_pi05_lora.bat: its environment,
dataset and single-camera assumptions differ from this experiment.

Dataset: D:/NERO/outputs/red_cube_base14_video_v1, existing train/val split.
Training-only q01/q99 statistics: configs/pi05_red_cube_quantiles.json,
240 Parquet files and 143932 training frames, 8 state/action dimensions.
Validation data must never be used to fit normalization.

Proposed baseline: base and wrist images, robot state and task text. No base14
conditioning in this first run. Freeze VLM and action-expert base weights;
adapt expert q_proj/v_proj using rank 8, alpha 16, dropout 0.05, plus existing
action and time projections. Chunk 50, n_action_steps 10, batch 1, bfloat16,
gradient checkpointing, zero loader workers. Hardware: RTX 4090 24 GB.
Actual peak memory and inference latency remain unmeasured.

Environment entry: setup_pi05_environment.cmd creates a separate environment
under D:/NERO/envs/pi05_lora and checks PI05 imports. It shares existing PyTorch
via system-site-packages; it is not a fully independent clone. Transformers
4.57.1 is a candidate pin, not yet validated with this vendored model.

Remaining implementation/verification:
1. Verify package imports and obtain a compatible pretrained PI05 checkpoint
   and PaliGemma tokenizer, recording source revision and retaining on D:.
2. Explicitly select base/wrist/state inputs and define camera-slot order for
   two cameras; existing specialized slot logic is for one wrist camera only.
3. Verify loading all base weights and refusing mismatches. Check adapter-only
   save/reload including normalization metadata and base-model reference.
4. Run actual training data through forward/backward, check finite nonzero LoRA
   gradients, measure memory, save/reload and compare deterministic predictions.
5. Run a short training test before a long run. Keep outputs separate from ACT.
6. Add an offline comparison using the same validation samples, 8 real action
   dimensions and padding masks, with repeatable flow-sampling noise. Check
   grasp approach/closure and inference latency before any deployment.

The PI05 forward currently masks padded time steps and slices to the 8 real
action dimensions. Its existing LoRA loader injects adapters after loading base
weights. These were inspected in code, not validated end-to-end in this session.
