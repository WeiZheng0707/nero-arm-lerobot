# NERO 项目交接文档

日期：2026-11-02（北京时间）

## 1. 当前任务与用户要求

当前准备从 ACT 对照实验转向 **π0.5 + LoRA**，使用已有红色方块入袋数据集。用户已经下载 π0.5 权重。用户要求先暂停推进，输出本交接文档到桌面。

- 用户在 Windows Anaconda Prompt 中执行命令，偏好完整单行命令，避免多行粘贴拼接错误。
- 数据集要求视频形式，数据、训练结果尽量放 D 盘。
- 用户关注真实抓到物块，不能把训练 loss 下降、环境输入有响应当作抓取成功。
- 当前没有确认任何模型已通过真实抓取验收。
- 不要擅自切换成 π0；用户最新选择为 π0.5 + LoRA。

## 2. 环境与关键路径

项目根目录：

    C:\Users\Administrator\Desktop\lerobot_nero_standalone_backup_20261019_before_changes\lerobot_nero_standalone_backup_20261019_before_changes

ACT 原环境 Python：

    D:\Anaconda3\envs\lerobot\python.exe

已安装的 π0.5 独立环境 Python：

    D:\NERO\envs\pi05_lora\Scripts\python.exe

硬件：NVIDIA RTX 4090，24 GB 显存。

π0.5 环境是通过 --system-site-packages 创建的 venv，复用 ACT 环境的 PyTorch，不是完整独立克隆。已安装 transformers 4.57.1、sentencepiece 0.2.2、tokenizers 0.22.2、huggingface-hub 0.36.2。

- π0.5 模块导入通过。
- lerobot.scripts.train 导入通过。
- pip 报告共享环境的 diffusers 0.40.0 要求 huggingface-hub >=1.23.0，与该 venv 的 0.36.2 冲突。当前 π0.5 导入未受影响，但不能宣称整个环境依赖一致。
- 原 ACT 环境中的 huggingface-hub 未卸载或修改。

## 3. 数据集与检测标注

原始数据：项目 outputs\nero_episodes。

独立标签：

    D:\NERO\outputs\base14_labels_full_v1

转换后视频数据集：

    D:\NERO\outputs\red_cube_base14_video_v1

- train：240 条轨迹，143932 帧，480 个视频。
- val：60 条轨迹，35979 帧，120 个视频。
- 两路相机 wrist/base，H.264 MP4，640×480，30 fps，CRF 20。
- 总占用约 2.7344 GiB，训练/验证轨迹无重叠。
- state/action 各 8 维：7 个关节（rad）+ 夹爪（0～1）。
- repo ID：local/nero_red_cube_base14_v1_train 与 local/nero_red_cube_base14_v1_val。
- 任务文字：Pick up the red cube and put it into the bag.
- 保留完整轨迹，没有裁成 480 帧；很多轨迹在 480 帧以后仍有运动。
- auto_end 不是成功标签，没有把这些轨迹全部确认为成功示范。

base14 检测配置：configs\base14_spread_candidate.json。

检测配置哈希：

    4041d853b98c73f30bc9c8a17f142dc6b73c6c8b9eec8aeac7e3706f7ba67544

base14 为方块/袋子的框、可见性、相对位置和面积比例，不包含 ball。部分遮挡/漏检仍存在，已经决定先做基线实验，不无限修改检测参数。

已核验 Parquet 的索引、有限数值、环境向量和 alignment.jsonl 一致，所有视频头帧数正确。转换器会完整解码核对视频帧数。真实样本解码和 ACT 前后向已通过。

## 4. ACT A/B 已完成 30000 步

A 输入：腕部图像 + 机械臂状态。
B 输入：腕部图像 + 机械臂状态 + base14。
两组都不将 base 图像作为 ACT 输入，虽然数据集保留该视频。

两组 seed=42、batch=8、chunk=60、n_action_steps=10、image/env dropout=0，先训练 2000，再恢复优化器与权重到总计 30000 步。

A 模型：

    D:\NERO\outputs\train\base14_a_2k_v1\checkpoints\030000\pretrained_model

B 模型：

    D:\NERO\outputs\train\base14_b_2k_v1\checkpoints\030000\pretrained_model

目录名 2k 不代表实际停留在 2000 步；training_step.json 均确认 30000。

验证使用同样 60 条轨迹，每条均匀取 8 个时刻，共 480 样本，排除末尾补齐动作。以下关节误差为 7 关节 MAE 的平均，夹爪为归一化 MAE：

| 模型 | 关节误差 rad | 夹爪误差 |
|---|---:|---:|
| A 2000 | 0.051380 | 0.125965 |
| B 2000 | 0.057220 | 0.126953 |
| A 30000 | 0.040123 | 0.077808 |
| B 30000 | 0.040811 | 0.086247 |
| 保持当前姿态基线 | 0.055260 | 0.217555 |

B 30000 的正确环境优于置换环境：置换后关节误差 0.045412（增加约 11.3%），夹爪 0.088384（增加约 2.5%）。但 B 总体仍未超过 A。B 在 28/60 条轨迹的关节误差优于 A，24/60 条夹爪误差优于 A。

前 10 个预测步的逐步误差均值：A 关节 0.018463、夹爪 0.049437；B 关节 0.018833、夹爪 0.047738。

这说明 B 对环境有依赖，不能证明真实移动物块后响应方向正确，更不能证明抓取成功。离线误差不是成功率。

报告：outputs\base14_a_2k_eval.json、base14_b_2k_eval.json、base14_a_30k_eval.json、base14_b_30k_eval.json、base14_30k_comparison.json。
评估入口：tools\evaluate_base14_policy.py。
续训脚本：train_base14_30k.cmd（恢复 last，到总计 30000；已完成，不需重复训练）。

## 5. 最近一次 ACT 部署故障与修复

用户在通道 0、v121 上启动 B 30000：双相机开启，从臂连接，回起始姿态成功，随后报 KeyError: 'ball'，steps sent=0。

原因：base14 无 ball 字段，但控制台日志和预览文字仍访问 scene['ball']。这发生在发送策略动作之前；回起始姿态运动已经发生，不能把 steps sent=0 解释成机械臂完全未动。

已修改 tools\nero_dual_cam_act_deploy.py：新增 scene_status，日志和预览共用，只在 ball 字段存在时显示。已做离线字段兼容和完整脚本语法检查；没有再次实机验证。

两路 RealSense 当次均报告 USB 2.1/USB 2.0 链路，640×480@30 RGB 带宽可能不稳。这不是 ball 异常直接原因，但需要另查实际帧龄和稳定性。

相机序列号：wrist=261622071323，base=261622078506。

## 6. 硬件状态与必须遵守的要求

最近退出日志为：UNCONFIRMED: gripper disable request was sent but feedback did not confirm disabled。
**夹爪失能没有得到新鲜反馈确认；当前硬件实际状态未知。** 不要写成已安全失能。

项目 AGENTS.md 要求：运动/使能前确认支撑、空夹爪及清空工作区；退出和失败清理须显式失能末端；通过推进的关节和末端反馈时间戳验证。只读检查不得使能/失能或改变模式。

历史通道曾为 0=主臂 v120、1=从臂 v121；用户最近指定通道 0 当从臂运行成功，但这不是永久映射，USB 变化后必须重新识别。不要盲目运行旧硬编码脚本。

当前部署 finally 会请求关节/夹爪失能，但不能宣称已自动完成新鲜反馈验证。Pi05 阶段只做离线工作，不操作硬件。

## 7. π0.5 用户提供的权重

目录（用户手动下载，保留原文件）：

    <项目根目录>\outputs\model

文件：config.json、model.safetensors、policy_preprocessor.json、policy_postprocessor.json、README.md、gitattributes。

- config.type=pi05，paligemma_variant=gemma_2b，action_expert_variant=gemma_300m。
- model.safetensors 为 14467165872 bytes，约 13.474 GiB；812 个 FP32 张量。
- safetensors 文件头和数据偏移长度一致，没有发现截断；未与官方 checksum 校验。
- 原配置 device=mps、dtype=float32、push_to_hub=true，不能直接沿用。准备配置覆盖为 cuda/bfloat16/不上传。
- 没有分词器文件，用户确认未下载到其他地方。

## 8. π0.5 本轮完成的代码准备

1. src\lerobot\policies\pi05\configuration_pi05.py 新增：
   - input_feature_keys（显式挑选输入）
   - tokenizer_name_or_path（支持本地分词器）
2. src\lerobot\policies\factory.py 的输入筛选从 ACT 扩展到 PI05。
3. src\lerobot\policies\pi05\modeling_pi05.py 使用配置指定分词器；双相机映射为 base、空左腕、wrist 顺序。
4. configs\pi05_red_cube_quantiles.json：只基于 train 的 q01/q99，8 维，无零宽度维度。
5. configs\train_pi05_dual_lora_smoke.json：准备的 200 步短测，尚未训练。
6. tools\train_pi05_dual.py：从配置创建策略，将 pretrained_path 设置为本地 outputs/model，并先校验本地分词器。
7. train_pi05_dual_smoke.cmd：调用独立环境，读取缓存设为 D:/NERO/cache/datasets。
8. setup_pi05_environment.cmd：安装入口，本轮已成功执行，不必重新安装。

短测配置：双相机+state+task，不消费 base14；冻结 VLM/动作专家基础参数，LoRA q/v，rank=8、alpha=16、dropout=0.05，同时训练 action/time 投影。batch=1、chunk=50、n_action_steps=10、bfloat16、gradient checkpointing、num_workers=0。

短测输出计划为 D:\NERO\outputs\train\pi05_dual_lora_smoke_200_v1；未开始训练。

已通过：配置解析、训练入口导入，以及用不同像素值验证相机顺序及 mask 为 [True,False,True]。
未通过/未执行：完整权重加载、真实前后向、优化器更新、适配器保存重载、推理速度测试。

## 9. 当前明确阻塞：权重键映射不兼容

实际尝试 PI05Policy.from_pretrained(outputs/model, strict=True) 已失败，不能把前面的“文件结构正确”说成“模型加载成功”。

加载器读到了全部 812 个张量并添加 model. 前缀，但视觉主干的键映射不匹配：

期望的例子：

    model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.weight

映射后出现的错误键路径例子：

    model.paligemma_with_expert.paligemma.model.vision_tower.encoder.layers.0...

报错包含大量 missing/unexpected keys。优先检查 modeling_pi05.py 中 _fix_pytorch_state_dict_keys，以及 transformers 4.57.1 下实际视觉模块结构。可能是已有重命名逻辑移除了 vision_model 层级，需要逐键/形状核实，不应直接 strict=False 跳过，更不能随机初始化视觉层后宣称在微调预训练模型。

本次加载过程较慢，结束时退出码 1。观察到 GPU 使用约 9746 MiB，但这不是训练峰值显存。测试进程已经结束。

## 10. 当前明确阻塞：分词器无法下载

需要 google/paligemma-3b-pt-224 分词器，目标目录 outputs\model\tokenizer。

所需文件：tokenizer.json、tokenizer_config.json、special_tokens_map.json，以及仓库提供的 tokenizer.model。无需重复下载 PaliGemma 大权重。

已获准尝试官方 snapshot_download（仅 tokenizer* 与 special_tokens_map.json），失败于 HTTPS SSL UNEXPECTED_EOF_WHILE_READING。随后 curl HEAD 也连接 huggingface.co:443 超时。尚未走到仓库认证检查，不能把当前网络问题说成已确认的权限拒绝。

官方地址：https://huggingface.co/google/paligemma-3b-pt-224/tree/main

不要关闭 TLS 验证或绕过访问限制。用户可浏览器下载，或提供可用网络/代理；仓库如要求接受条款，需用户自行完成。现有 Pi05PromptBuilder 使用 local_files_only=True。

## 11. 恢复工作的推荐顺序

1. 先修复并测试权重键映射，完整严格加载用户提供的模型；不需要等 tokenizer 才能查权重问题。
2. 获取/验证本地 tokenizer 后，跑真实训练集样本，检查 task/state 提示词、8 维归一化和双相机顺序。
3. 确认 LoRA 数量、冻结参数、有限非零梯度和 optimizer 更新；测显存。
4. 完成保存/重载一致性：适配器依赖基础权重和归一化文件，路径必须可解析；当前代码依赖外部 quantile_stats_path，需考虑保存时携带统计文件。
5. 执行 200 步短测并检查保存结果，随后再准备长训练命令；短测的 200 步 scheduler 是排错用途，正式训练要重新设定，不可直接当成 30000 步计划。
6. 为 PI05 做与 ACT 相同验证集的独立评估，控制随机流采样，屏蔽补齐帧，分开关节/夹爪误差。
7. 实测推理延迟与接近/夹持阶段，再讨论部署。现有 ACT 部署脚本不能直接加载 PI05。

本交接时：没有启动 π0.5 正式训练，没有发送机械臂动作，没有验证新的夹爪失能状态。
