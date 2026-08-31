# Changelog

## [0.5.0] - 2026-08-03

- 新增独立的 host high-level-policy sim2real 运行时：使用严格的 msgpack/ZeroMQ 协议、异步 receding-horizon replanning、时间戳对齐调度，以及 50 Hz 输出安全校验和限速。
- 扩展 G1 外设支持：加入 OpenNeck 0.2.0 物理角度控制、Pico HMD 主动视觉映射、LinkerHand O6 somehand 0.3.0 手势控制，以及手部和颈部状态回读。
- 更新 sim2real 录制与审阅流程：采用 `schema.json`、`episodes.jsonl`、逐 episode HDF5 和压缩 MP4 布局，记录可选手部/颈部状态与动作，并新增同步 recording viewer。
- 新增匹配的 G1 模型/策略组合：默认 `g1_29dof.xml` 配合 `ckpt/track_g1.{pt,onnx}`，neck-and-O6 版本配合 `g1_29dof_neck_o6.xml` 和 `ckpt/track_g1_neck_o6.{pt,onnx}`。
- 更新 OmniXtreme-style benchmark，并增强 Pico/RealSense 故障恢复、GMR mocap-entry cold start、high-level-policy watchdog 和引用安全处理。

### 迁移说明

- v0.4 的根目录 `track.{pt,onnx}` 路径已替换为 `ckpt/track_g1.{pt,onnx}`；neck-and-O6 运行时必须使用对应的模型和策略组合。
- 旧的 attribute-based sim2real HDF5 格式不再支持；录制、转换和审阅工具使用当前 manifest-based source layout。
- Host-policy 网络协议不提供旧 envelope 兼容，OpenNeck 旧 normalized API 也不再支持；Teleopit 与 companion runtime 必须使用匹配版本。

## [0.4.0] - 2026-06-25

- 改进 Pico 实时控制：支持 pico-bridge 0.2.1、`ARMS` 模式，以及保留 retargeter warm-start 的模式切换/暂停恢复。
- 新增可选 LinkerHand L6/O6 sim2real 控制，支持 Pico gripper 输入和低延迟 L6 `vr_hand_pose`。
- 新增 Pico sim2real 手动 HDF5 录制，以及用于训练数据采集的交互式 Pico motion recorder。
- 优化训练数据流程：minimal HDF5 shards、显式 precompute、rewind 采样和更新后的 tracking rewards。

## [0.3.0] - 2026-05-12

- 重构实时输入栈，Pico 4 统一使用 pico-bridge 0.2.0 in-process receiver，并移除旧 ZMQ/onboard Pico 路径。
- 统一 sim/sim2real 实时 reference buffer、pause/resume realignment 与速度平滑逻辑。
- 扩展 UDP BVH、online sim、多 viewer 与固定相机支持。
- 拆分 sim2real reference/safety 运行时模块，并更新 G1 MuJoCo 相机资产。

## [0.2.0] - 2026-04-03

- 接入 Pico 4 遥操作与 G1 Bridge SDK。
- 新增独立 Standing 控制器、离线播放键盘控制与 Pico sim2sim 模式控制。
- 优化实时 mocap 缓冲与 catch-up，并将发布模型升级至 30k checkpoint。

## [0.1.1] - 2026-03-28

- 数据集改为 shard-only 输出。
- 引入外部资源管理并瘦身仓库。

## [0.1.0] - 2026-03-25

- 首个公开版本。
- 支持 General-Tracking-G1 全身追踪训练与 ONNX sim2sim 推理。
- 支持 Pico 4 VR 遥操作与 Unitree G1 真机部署。
