#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("nero_follower")
@dataclass
class NeroFollowerConfig(RobotConfig):
    """松灵 NERO **从臂**配置：7 个关节 + 1 个夹爪 = 8 轴。

    写法刻意对齐 `PIPERFollowerConfig`：只放"确实需要按机器改"的几个量，
    其余全走默认。`id` 和 `calibration_dir` 由 `RobotConfig` 提供，不用重写。

    连接参数（None = 按**角色**自动选，从臂 -> "0" / "can0"）：
      - Windows : `interface="agx_cando"`, `channel="0"`
      - Linux   : `interface="socketcan"`, `channel="can0"`

    ⚠️ 主臂侧会自动选 **"1" / "can1"**（两边刻意错开，对齐 piper 的
    can_follower / can_master 思路），但 Windows 上 channel 是 cando 的
    **设备序号**、会随插拔顺序变，所以**建议两边都显式指定**。
    """

    # ---- 相机 ----
    # 键是**裸相机名**（如 "front"），全仓库统一这个约定：
    #   - observation_features / get_observation() 用裸名 —— 上游
    #     hw_to_dataset_features() 和 build_dataset_frame() 会自己补/剥
    #     `observation.images.` 前缀；
    #   - 只有 camera_features 那一处需要带前缀（get_features_from_robot()
    #     是原样并入的，不再补）。
    # 两处约定不同，搞反了会得到 observation.images.observation.images.xxx。
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # ---- CAN 连接 ----
    interface: str | None = None
    # 留 None 时**按角色自动选**：从臂 -> "0"（Linux 是 "can0"）。
    # 主臂侧会自动选 "1" / "can1"，两边刻意错开，免得主从抢同一条总线。
    # 对齐 piper：它是把 follower/master 写死成两个不同的名字（不给配置项）。
    # ⚠️ 仍然**强烈建议显式指定**：Windows 上 channel 是 cando 枚举到的
    #    **设备序号**，会随插拔顺序变；写死成固定的值最好。
    channel: str | None = None
    # 固件档位（"default" / "v111" / "v112" / "v120" / "v121"）。
    # None = 交给 SDK 自己解析 —— 实测 pyAgxArm 会读出真实固件并把档位纠正过来
    # （传 default 时它自己换成 v120），所以**推荐留 None**。
    firmware: str | None = None

    # ---- 安全 ----
    # 单次下发相对**当前实测位姿**的最大变化量（rad）。
    # 两台臂各自独立标零，静止读数能差 100°+，没有限幅 = 上电就甩臂。
    # None = 不钳制（不建议）。
    max_relative_target: float | None = 0.05
    # 使能超时（秒）。⚠️ 实测状态回读滞后可超过 15 秒（低速帧只有 1 Hz 量级），
    # 给小了会把"还没刷出来"误报成"使能失败"。
    enable_timeout: float = 25.0
    # 运行速度百分比 [0, 100]
    speed_percent: int = 100

    # ---- 夹爪（第 8 轴）----
    # True  = 本从臂接管夹爪，维度 **8**（joint_1..joint_7 + gripper）
    # False = 不装夹爪，维度 **7**（joint_1..joint_7）
    #
    # ⚠️ 这是**静态决定**，连接前后必须一致 —— 不要在连接时再变。
    #    原因：`record.py` 的次序是「先读 features 建数据集」-> `robot.connect()`，
    #    所以 features 在**连接之前**就被拿去建数据集了。若维度在连接时才变，
    #    会出现"数据集声明 8 列、`get_observation()` 只给 7 列"，
    #    录制时 `build_dataset_frame()` 直接 KeyError（而且是在录的时候才炸）。
    #    这也是 `NeroMotorsBus._init_gripper()` 失败**直接报错**而不降级的原因。
    #
    # ⚠️ 遥操作时**主从两侧必须一致**（`NeroLeaderConfig.use_gripper` 要相同），
    #    否则 action 的键对不上，从臂 `send_action()` 会 KeyError。
    use_gripper: bool = True
    gripper_mode: str = "width"
    gripper_max_width_m: float = 0.07
    gripper_max_angle_deg: float = 100.0
    gripper_force_n: float = 1.0
    gripper_send_threshold: float = 0.005
    disable_gripper_on_disconnect: bool = True
    feedback_timeout_s: float = 0.5

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.gripper_mode not in ("width", "angle"):
            raise ValueError("gripper_mode must be 'width' or 'angle'")
        for name in ("enable_timeout", "feedback_timeout_s", "gripper_max_width_m",
                     "gripper_max_angle_deg", "gripper_force_n"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 <= self.gripper_send_threshold <= 1:
            raise ValueError("gripper_send_threshold must be in [0, 1]")
        if self.max_relative_target is not None and not math.isfinite(self.max_relative_target):
            raise ValueError("max_relative_target must be finite")
        if self.max_relative_target is not None and self.max_relative_target < 0:
            raise ValueError(
                f"max_relative_target 必须 >= 0（或 None 表示不钳制），得到 {self.max_relative_target}"
            )
        if self.enable_timeout <= 0:
            raise ValueError(f"enable_timeout 必须为正数，得到 {self.enable_timeout}")
        if not 0 <= self.speed_percent <= 100:
            raise ValueError(f"speed_percent 必须在 [0, 100] 内，得到 {self.speed_percent}")
