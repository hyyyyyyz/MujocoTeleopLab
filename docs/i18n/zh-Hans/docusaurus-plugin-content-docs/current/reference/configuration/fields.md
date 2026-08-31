---
sidebar_position: 2
---

# 配置字段

本页列出 Teleopit 的全部 Hydra 配置字段。

## 顶层字段

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `policy_hz` | int | — | 策略推理频率（Hz） |
| `pd_hz` | int | `200` | PD 控制器频率（Hz，仅仿真），通常高于 `policy_hz` |
| `viewers` | str/list | `sim2sim` | 可视化窗口集合：`mocap`、`retarget`、`sim2sim`、`camera`、`all`、`none`。`all` 打开 `mocap`、`retarget` 和 `sim2sim`；如需相机画面需显式加入 `camera` |
| `realtime` | bool | `false` | 是否启用实时模式（实机部署时需开启） |
| `num_steps` | int | — | 仿真总步数；设为 `-1` 表示无限运行 |
| `keyboard.enabled` | bool | `false` | 是否启用 sim2sim 实时键盘模式控制 |
| `playback.pause_on_end` | bool | `false` | 回放结束后是否暂停（而非退出） |
| `playback.keyboard.enabled` | bool | `false` | 是否启用键盘控制回放进度 |

## Robot 字段

机器人相关配置位于 `robot/` 子目录。以 `robot/g1.yaml` 为例：

| 字段 | 类型 | 说明 |
|---|---|---|
| `type` | str | 写入录制 schema 的稳定机器人类型，G1 为 `unitree_g1_29dof` |
| `num_actions` | int | 策略输出的动作维度（即受控关节数） |
| `xml_path` | str | MuJoCo MJCF 模型文件路径 |
| `d435i_rgb` | camera | G1 MJCF 中的固定 RGB 相机；配合 `viewers=[sim2sim,camera]` 显示画面 |
| `kps` | list[float] | 各关节的比例增益（P 增益） |
| `kds` | list[float] | 各关节的微分增益（D 增益） |
| `default_angles` | list[float] | 默认关节角度（弧度），也是策略动作的零点 |
| `torque_limits` | list[float] | 各关节的力矩上限 |

## Controller 字段

控制器配置位于 `controller/` 子目录。

| 字段 | 类型 | 说明 |
|---|---|---|
| `policy_path` | str | **必填。** 策略模型文件路径（ONNX 格式） |
| `device` | str | 推理设备，如 `"cpu"` 或 `"cuda:0"` |
| `action_scale` | float | 动作缩放系数 |
| `clip_range` | list[float] | 动作裁剪范围，格式为 `[min, max]` |
| `default_dof_pos` | list[float] | 默认关节位置，用于计算控制目标 |

### 关键说明：`default_dof_pos` 与动作计算

策略输出的 action 是相对于 `default_dof_pos` 的**偏移量**，最终的关节控制目标按如下公式计算：

```
target = clip(action, clip_range) * action_scale + default_dof_pos
```

因此，`default_dof_pos` 决定了策略输出的"零点"。如果该值与训练时使用的不一致，策略的行为将完全偏离预期。

## Input 字段

输入源配置位于 `input/` 子目录，不同输入源的字段各异。

### BVH 输入（`input/bvh.yaml`）

| 字段 | 类型 | 说明 |
|---|---|---|
| `bvh_file` | str | BVH 文件路径 |
| `bvh_format` | str | BVH 骨骼格式标识 |
| `human_format` | str | 人体骨架格式 |

> BVH 输入不设置 `input.provider` — 由配置组名自动推断。

### Pico 4 输入（`input/pico4.yaml`）

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `provider` | str | `pico4` | 输入源类型 |
| `human_format` | str | `pico_bridge` | 重定向骨架格式 |
| `pico4_timeout` | float | `60` | 等待设备连接的超时时间（秒） |
| `pico4_buffer_size` | int | `60` | 帧缓冲区大小 |
| `pause_button` | str | `A` | 用于暂停/恢复的手柄按钮名称 |
| `pause_debounce_s` | float | `0.25` | 暂停按钮防抖时间 |
| `arms_button` | str | `B` | Pico 中用于切换 `MOCAP` / `ARMS` 的按钮 |
| `arms_debounce_s` | float | `0.25` | 双臂模式按钮防抖时间 |
| `bridge_host` | str | `0.0.0.0` | Teleopit host receiver 绑定地址 |
| `bridge_port` | int | `63901` | Teleopit host receiver TCP/UDP 端口 |
| `bridge_discovery` | bool | `true` | 是否启用 pico-bridge 发现广播 |
| `bridge_advertise_ip` | str/null | `null` | 可选的 host 广播 IP 覆盖 |
| `bridge_start_timeout` | float | `10.0` | 启动 bridge 的超时时间 |
| `bridge_history_size` | int | `120` | bridge 保留的 Pico 帧历史长度 |
| `video.enabled` | bool | `false` | 通过 pico-bridge 0.2.1 将 host 相机预览发送回 Pico |
| `video.source` | str/null | `null` | 视频源：`mujoco`、`realsense` 或 `test-pattern` |
| `video.width` / `height` / `fps` | int | `1280` / `720` / `30` | 视频采集/渲染设置 |
| `video.device` | str/null | `null` | 可选的 RealSense 序列号 |

## Realtime 字段

实时模式相关字段，仅在 `realtime=true` 时生效。

| 字段 | 说明 |
|---|---|
| `retarget_buffer_enabled` | 是否启用重定向缓冲 |
| `retarget_buffer_window_s` | 缓冲窗口大小 |
| `retarget_buffer_delay_s` | 缓冲延迟 |
| `reference_steps` | 参考轨迹窗口步数 |
| `realtime_buffer_warmup_steps` | 播放前预热帧数 |
| `reference_velocity_smoothing_alpha` | 速度平滑系数 |
| `reference_anchor_velocity_smoothing_alpha` | 锚点速度平滑系数 |

## Sim2Real 字段

以下字段用于 sim2real 配置（`sim2real.yaml`、`pico4_sim2real.yaml`）。

sim2real 默认使用 `viewers=none`。设置 `viewers=retarget` 可打开一个可选的
MuJoCo 窗口显示重定向参考；`sim2sim`、`mocap`、`camera` 和 `all`
仅用于仿真 viewer。

### 安全相关

| 字段 | 说明 | 默认值 |
|---|---|---|
| `startup_ramp_duration` | 进入 `STANDING` 后的 Kp ramp 时长；逐步提高 PD 增益，不改变 policy target | `2.0` |
| `joint_vel_limit` | 关节速度限制（rad/s），超过时触发急停 | `10.0` |
| `mocap_switch.check_frames` | 切换到 MOCAP 前所需的连续有效帧数 | `10` |
| `arm_mocap.controlled_joint_indices` | Pico `ARMS` 模式下由实时 retargeting 驱动的 G1 关节 | `[15..28]` |

### 主机 High-Level Policy（独立 sim2real）

`high_level_policy_sim2real.yaml` 只供
`scripts/run/run_high_level_policy_sim2real.py` 使用。它会启动 camera、network
client、robot-control、LinkerHand O6 和 OpenNeck worker；不会启动 PicoBridge、GMR
或 retarget reference worker。主机 LeRobot 环境保持独立，并且必须跟随当前
client/server 消息结构与协议测试。唯一共享的数据文件是 `hand_calibration.json`。

| 字段 | 说明 | 默认值 |
|------|------|--------|
| `camera.source` | Onboard 策略相机：`realsense`，或仅供集成测试的 `test-pattern` | `realsense` |
| `camera.width` / `height` / `fps` | 精确的策略图像契约 | `640` / `480` / `30` |
| `camera.device` | 可选 RealSense 序列号 | `null` |
| `standing_return_ramp_duration` | 从主动控制返回 `STANDING` 时的 Kp ramp 时长 | `2.0` |
| `high_level_policy.endpoint` | 主机策略 ZeroMQ TCP endpoint | `tcp://127.0.0.1:5555` |
| `high_level_policy.task` | reset 和每个 observation 都会发送的非空任务 prompt | `demo` |
| `high_level_policy.timeout_s` | 单次网络请求 deadline；超时会暂停 `POLICY` | `1.0` |
| `high_level_policy.reconnect_backoff_s` | 建立新 session 时的重试间隔 | `1.0` |
| `high_level_policy.replan_steps` | 两次请求之间的最小 30 Hz source-frame 间隔；不得超过主机报告的 horizon | `3` |
| `high_level_policy.jpeg_quality` | 640x480 RGB 帧的 JPEG 质量 | `90` |
| `high_level_policy.max_observation_age_s` | 跳过请求前允许的最大 camera/observation age | `0.15` |
| `high_level_policy.max_result_age_s` | 拒绝已接收结果前允许的最大本地 IPC age | `0.1` |
| `high_level_policy.entry_timeout_s` | 建立 entry session 并收到其第一份有效 chunk 的最长时间，以及恢复时等待新鲜 chunk 的最长时间 | `5.0` |
| `high_level_policy.hold_s` | active plan horizon 结束后、action watchdog 暂停 `POLICY` 前保持最终 reference 的 grace period | `3.0` |
| `high_level_policy.safety.root_height_min_m` / `root_height_max_m` | 可接受的绝对 root 高度范围 | `0.55` / `1.05` |
| `high_level_policy.safety.max_root_xy_speed_m_s` | 应用于 50 Hz scheduler 输出的 root XY 速度限制 | `2.5` |
| `high_level_policy.safety.max_root_displacement_m` | 50 Hz 输出 rate limiter 使用的 source-frame 等效 3D root 步长 | `0.1` |
| `high_level_policy.safety.max_yaw_rate_rad_s` | 应用于 50 Hz scheduler 输出的 root yaw rate 限制 | `2.5` |
| `high_level_policy.safety.max_joint_rate_rad_s` | 应用于 50 Hz scheduler 输出的单关节 rate 限制 | `10.0` |
| `high_level_policy.safety.max_joint_projection_rad` | 将 G1 关节 reference 裁剪到位置限位时允许的最大修正量 | `0.1` |
| `high_level_policy.safety.neck_yaw_min_deg` / `neck_yaw_max_deg` | OpenNeck yaw 裁剪范围 | `-45` / `45` |
| `high_level_policy.safety.neck_pitch_min_deg` / `neck_pitch_max_deg` | OpenNeck pitch 裁剪范围 | `-40` / `40` |

请求循环采用异步 receding-horizon。隔离的 client 最多只有一个 ZeroMQ 请求在途，
按照配置的 source-frame stride 选择最新的合格 observation，并在主机推理期间继续执行
当前 action plan。较新的 response 会依据其中回显的 onboard 单调 observation 时间戳
替换该计划。

当所需修正量不超过 `high_level_policy.safety.max_joint_projection_rad` 时，G1 reference
joint position 会裁剪到 `real_robot.joint_pos_lower/upper`；更大的修正量会导致 chunk 被拒绝。
OpenNeck yaw/pitch 会裁剪到配置范围，单纯的 neck 越界不会导致 chunk 被拒绝。由于 canonical
50D action 的所有字段都处于启用状态，初始运行时要求
`hands.driver=linkerhand_o6`、左右两只手以及 `neck.driver=openneck`。OpenNeck 策略值
在 onboard 裁剪并完成 chunk 验证后直接发送给 `move_deg(yaw, pitch)`；不会应用 Pico
dead-zone 或 pitch-gain 映射。

### 真机 SDK

| 字段 | 说明 | 默认值 |
|---|---|---|
| `real_robot.network_interface` | Unitree DDS 通信网络接口。PC 通过网线连接 G1 控制时，用 `ifconfig` 找到这根网线对应的接口名并填写，例如 `enp130s0`；在机器人 onboard 计算机上运行时通常使用 `eth0` | `eth0` |
| `real_robot.kp_real` | 真机比例增益（各关节） | — |
| `real_robot.kd_real` | 真机微分增益（各关节） | — |
| `real_robot.kd_damping` | 阻尼模式 kd | `8.0` |
| `real_robot.control_mode` | 踝关节控制模式（`PR` = Pitch-Roll） | `PR` |
| `real_robot.joint_pos_lower` | 关节位置下限（rad） | — |
| `real_robot.joint_pos_upper` | 关节位置上限（rad） | — |

### 暂停/恢复（Pico sim2real）

实时 Pico 恢复追踪时会先重新居中航向和地面平面位置。操作者应保持静止，并尽量贴近暂停时的姿态，以减少参考突变。

### 灵巧手（Pico sim2real）

`hands.enabled=true` 要求 `input.provider=pico4`，并以本地 editable 方式安装
`third_party/linkerhand-python-sdk` 和 `third_party/somehand`。启用后，手控会在所有 sim2real 模式中保持生效。
`gripper` 支持 `linkerhand_l6` 和 `linkerhand_o6`。对应手柄侧面的握持扳机键（grip）
是安全使能键：保持按住时，食指扳机键（trigger）会在配置的张开和闭合姿态之间插值；
松开侧面握持扳机键会让该侧手张开。
`vr_hand_pose` 支持 `linkerhand_l6` 和 `linkerhand_o6`：手部 pose 消失时，对应侧会保持上一条命令；
所选手的速度会设为最大值；Teleopit 会先将 Pico 手部状态转成 21 个 landmarks，
再只通过 somehand 0.3.0 公开的 `somehand.api` 调用。

| 字段 | 说明 | 默认值 |
|---|---|---|
| `hands.enabled` | 启用可选手部运行时 | `false` |
| `hands.mode` | `gripper` 或 `vr_hand_pose` | `gripper` |
| `hands.driver` | 手部设备驱动：`linkerhand_l6` 或 `linkerhand_o6` | `linkerhand_l6` |
| `hands.sides` | 控制侧 | `[left, right]` |
| `hands.rate_hz` | gripper 最大命令频率（Hz） | `30.0` |
| `hands.frame_timeout_s` | 手柄或手部 pose 过期阈值 | `0.3` |
| `hands.linkerhand_l6.left_can` / `right_can` | 左右手 CAN 通道 | `can0` / `can1` |
| `hands.linkerhand_l6.speed` | `gripper` 使用的 L6 速度；`vr_hand_pose` 会覆盖为最大速度 | 见配置 |
| `hands.linkerhand_l6.deadman_threshold` | 启用单侧控制所需的最小 grip 值 | `0.5` |
| `hands.linkerhand_l6.trigger_deadzone` | trigger 两端死区 | `0.05` |
| `hands.linkerhand_l6.open_pose` / `close_pose` | L6 的 6 维张开/闭合姿态 | 见配置 |
| `hands.linkerhand_o6.left_can` / `right_can` | 左右 O6 手 CAN 通道 | `can0` / `can1` |
| `hands.linkerhand_o6.speed` | `gripper` 使用的 O6 速度；`vr_hand_pose` 会覆盖为最大速度 | 见配置 |
| `hands.linkerhand_o6.open_pose` / `close_pose` | O6 的 6 维张开/闭合姿态 | 见配置 |
| `hands.somehand.l6_config_path` | L6 `vr_hand_pose` 使用的 somehand 0.3.0 官方双手 L6 配置 | 见配置 |
| `hands.somehand.o6_config_path` | O6 `vr_hand_pose` 使用的 somehand 0.3.0 官方双手 O6 配置 | 见配置 |
| `hands.somehand.rate_hz` | 低延时 `vr_hand_pose` 命令频率（Hz） | `60.0` |
| `hands.somehand.max_iterations` | `vr_hand_pose` 的 somehand solver 迭代上限 | `12` |
| `hands.somehand.temporal_filter_alpha` | somehand 输入 landmarks 平滑 alpha；`1.0` 表示关闭平滑延时 | `1.0` |
| `hands.somehand.output_alpha` | somehand qpos 输出平滑 alpha；`1.0` 表示关闭平滑延时 | `1.0` |

### OpenNeck 主动视觉（Pico sim2real）

`neck.enabled=true` 要求 `input.provider=pico4` 和 `openneck` extra。neck worker
复用 Teleopit 已有的 Pico receiver，不会启动第二个 `PicoBridge` 或 RealSense 管线。
OpenNeck 作为非关键 sim2real worker 运行，不会改变策略观测。头部运动来自独立的头显
`PicoFrame.head.rotation`，并相对于同一个源帧中的 `Body.Spine3` 进行映射。neck 路径
绝不读取全身动捕的 `Body.Head` 骨架关节；人体模型约束可能使该关节低估极端低头角度。
头显姿态更新不受 body 重复帧过滤影响。mapper 使用固定的 PICO 中立姿态且不进行颈部侧
EMA；启动时不会把操作者的第一帧姿态采集为新的零位，因此开始追踪时操作者不需要保持头部
朝正前方。Teleopit 将受支持的 PICO 约定转换为 OpenNeck 的物理约定——正 yaw 向左转，
正 pitch 向上看。对原始相对角度应用 `neck.dead_zone_deg` 后，Teleopit 将 pitch 乘以
`neck.pitch_gain`（默认 `1.4`），而 yaw 仍保持一比一。随后通过 OpenNeck 0.2.0 的
`move_deg()` 发送得到的物理角度。OpenNeck 负责直驱角度到舵机步数的转换，并将每个目标
裁剪到标定文件中的机械步数限位。

OpenNeck 0.2.0 标定文件使用 `yaw_center_step`、`yaw_min_step`、
`yaw_max_step` 和 `yaw_step_sign` 等角度控制字段（pitch 使用对应字段）。不支持以前的
OpenNeck 归一化配置；运行 `openneck calibrate` 创建当前格式的文件。Teleopit 已移除的
`neck.yaw_range_deg`、`neck.pitch_range_deg` 和 `neck.invert_*` 键会被拒绝，
而不是被忽略。

| 字段 | 说明 | 默认值 |
|---|---|---|
| `neck.enabled` | 启用可选 OpenNeck worker | `false` |
| `neck.driver` | 头颈设备驱动插件；当前为 `openneck` | `openneck` |
| `neck.config_path` | 可选 OpenNeck 0.2.0 角度标定配置路径 | `null` |
| `neck.port` | 可选串口覆盖，例如 `/dev/ttyACM0` | `null` |
| `neck.rate_hz` | 最大头颈命令频率（Hz） | `60.0` |
| `neck.frame_timeout_s` | Pico 头显/Spine3 姿态过期阈值 | `0.2` |
| `neck.active_modes` | 允许头颈运动的 sim2real 模式 | `[standing, mocap, arms, pause]` |
| `neck.dead_zone_deg` | yaw/pitch 死区（度） | `0.5` |
| `neck.pitch_gain` | 死区后应用于头显相对 pitch 的增益 | `1.4` |
| `neck.center_on_start` / `center_on_shutdown` | worker 启动/关闭时回中云台 | `true` / `false` |
| `neck.release_on_shutdown` | 关闭后在支持时释放舵机扭矩 | `false` |
| `neck.dry_run` | 只计算命令，不打开 OpenNeck 硬件 | `false` |

### HDF5 录制（Pico sim2real）

`recording.enabled=true` 只支持 `input.provider=pico4`、
`input.video.enabled=true`、`input.video.source=realsense`，并且需要交互式终端。
录制是手动控制：`R` 开始 episode，`S` 保存当前 episode，`D` 丢弃当前 episode，
`Q` 关闭。可以录制 `STANDING`、`MOCAP`、`ARMS` 和暂停状态的 mocap。

`sim2real_record.yaml` 会同时启用录制和必需的 RealSense `input.video`
路径。录制不会打开第二路相机，而是消费 `pico_input` 已经产生的同一批帧。

| 字段 | 说明 | 默认值 |
|---|---|---|
| `recording.enabled` | 启用手动 HDF5 录制 | `false` |
| `recording.output_dir` | 数据集根目录 | `data/recordings/sim2real_hdf5` |
| `recording.task` | 写入 `episodes.jsonl` 的 episode 任务 prompt | `demo` |
| `recording.fps` | 录制/视频主时钟频率 | `30` |
| `recording.min_episode_seconds` | 保存时短于该时长的 episode 会被丢弃 | `1.0` |
| `recording.record_modes` | 允许开始录制和写帧的模式 | `[standing, mocap, arms, pause]` |
| `recording.camera.key` | RGB 图像数据集 key | `observation.images.d435i_rgb` |
| `recording.camera.width` / `height` / `fps` | RealSense RGB 采集设置 | `640` / `480` / `30` |
| `recording.camera.device` | 可选 RealSense 序列号 | `null` |
| `recording.video.codec` / `quality` / `pixelformat` | MP4 sidecar 编码设置 | `libx264` / `8` / `yuv420p` |

RealSense 帧超时或断连时会在后台重建采集 pipeline，绝不会停止 Pico 输入或 G1
控制。按 `R` 开始录制前必须存在新鲜相机帧。录制期间一秒内没有新鲜帧时，当前
episode 会被丢弃；相机恢复后录制仍保持空闲，直到操作员再次按 `R`。
如果整个 `pico_input` worker 退出，`robot_control` 会继续运行并保持最新命令；
Unitree 遥控器仍可用于返回 `STANDING` 或请求 `DAMPING`。

录制器会创建一份便于编辑的源数据集：

```text
recording.output_dir/
├── schema.json
├── episodes.jsonl
├── data/
│   └── episode_000000.h5
└── videos/
    └── d435i_rgb/
        └── episode_000000.mp4
```

`schema.json` 保存 FPS、`robot_type`、`hand_type`、`neck_type` 和 feature 定义。
`robot_type` 来自 `robot.type`；未启用灵巧手时 `hand_type` 为 `none`，否则为
配置的 `hands.driver`。未启用主动视觉颈部控制时 `neck_type` 为 `none`，否则为
配置的 `neck.driver`。这些 enabled 标志直接决定是否录制对应的 state 和 action
字段；没有单独的录制开关。`episodes.jsonl` 每行对应一个已保存的 episode，包含
`episode_index`、`frames`、可编辑的 `task`、HDF5 路径和视频路径。因此修改任务
prompt 不需要重写 HDF5 或 MP4。使用相同 schema 再次启动录制时，会从下一个
episode index 继续追加，并且可以使用不同的 `recording.task`。

该格式有意不兼容之前依赖 HDF5 根属性的布局。请使用空的
`recording.output_dir`；如果已有 schema 不匹配，录制 worker 会拒绝该数据集并
退出，不写入 episode。录制属于非关键进程，因此 sim2real 主控制运行时会继续
运行并报告 worker 故障。在 `episodes.jsonl` 条目提交前中断的 episode 会在下次
录制 worker 启动时被丢弃，并且不会占用 episode index。

HDF5 datasets：

```text
frame_index                    int64[N]
timestamp                      float64[N]
observation.state              float32[N, 68]
observation.state.hand         float32[N, 12]  # 仅启用灵巧手时存在
observation.state.neck         float32[N, 2]   # 仅启用 OpenNeck 时存在
observation.mode               int8[N]
action                         float32[N, 36]
action.hand                    float32[N, 12]  # 仅启用灵巧手时存在
action.neck                    float32[N, 2]   # 仅启用 OpenNeck 时存在
```

HDF5 文件仅包含上述逐帧数组，不保存录制元数据根属性。RGB 帧保存在 MP4 中，
并通过 `episodes.jsonl` 与 episode 关联；不会写入原始 RGB HDF5 dataset。

`observation.state` 的顺序是 `joint_pos(29)`、`joint_vel(29)`、
`base_quat_wxyz(4)`、`base_ang_vel(3)` 和 `projected_gravity(3)`。
`observation.state.hand` 是最新的 LinkerHand 硬件回读：
`left_state(6) + right_state(6)`，使用 SDK 的 0-255 关节数值。
`observation.state.neck` 是 OpenNeck `read_deg()` 返回的最新舵机位置：
以度为单位的 `[yaw_deg, pitch_deg]`。
`observation.mode` 是数值类别：`standing=0`、`mocap=1`、
`arms=2`、`pause=3`。`action` 是当前 reference qpos：
`root_pos(3) + root_quat_wxyz(4) + reference_joint_pos(29)`。它是 motion tracker
消费的高层参考，不是 tracker policy 的原始输出，也不是最终下发给 G1 的关节目标。
`action.hand` 是手部 worker 最新的 LinkerHand 命令：
`left_pose(6) + right_pose(6)`，使用 SDK 的 0-255 pose 数值。
`action.neck` 是 OpenNeck 成功执行命令后返回的最新机械限位裁剪目标：以度为单位的
`[yaw_deg, pitch_deg]`。正 yaw 向左转，正 pitch 向上看。可达范围来自 OpenNeck
标定文件，因此录制 schema 中没有固定范围。
