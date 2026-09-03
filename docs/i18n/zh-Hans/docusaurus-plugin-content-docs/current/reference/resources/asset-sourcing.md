---
sidebar_position: 2
---

# 额外仿真资产来源

新增桌面物体或任务场景时，优先选择具有明确资产许可证和稳定版本的来源仓库。开始适配前，
先在 `assets/manifests/` 中记录来源。主仓库只保留清单和适配代码；下载后的 mesh、纹理和大型
归档继续放在被忽略的本地路径。

## 候选来源

| 来源 | 提供内容 | 许可证证据 | 对本项目的适配情况 |
| --- | --- | --- | --- |
| [robosuite](https://github.com/ARISE-Initiative/robosuite) | MuJoCo 场景、primitive 生成器、瓶子、罐头、面包、麦片、牛奶、柠檬、盘子、门 | 仓库 `LICENSE` 为 MIT；同时单独说明所含 MuJoCo 代码为 Apache-2.0。再分发前仍需逐个检查资产目录。 | 适合作为第一批小型、独立物体适配来源。其 XML 是可组合物体，不是现有 43 自由度 G1 场景的直接替换。 |
| [RoboCasa](https://github.com/robocasa/robocasa) | 厨房设施、3200 多个物体、2500 多个场景和日常任务定义 | 上游 README 说明代码为 MIT，资产和数据集为 CC BY 4.0。 | 适合后续厨房任务；下载量约 10 GB，需要转换层和署名清单。 |
| [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) | 高质量机器人模型和示例场景 | 每个机器人目录有独立 `LICENSE`；README 展示了 BSD-3-Clause、Apache-2.0、MIT 等不同许可证。 | 适合机器人/模型参考，不是第一批桌面物体包的首选。不能假设顶层许可证覆盖机器人子目录。 |
| [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) | MuJoCo 运动和操作环境代码及集成 | 上游 README 说明仓库内容为 Apache-2.0，Polyhaven 粗糙地面纹理为 CC0。 | 适合作为环境模式和随机化参考，不是现成的 G1/Dex3 物体库。 |
| [ManiSkill](https://github.com/haosulab/ManiSkill) | 大型操作基准和刚体环境 | 上游 README 说明环境使用宽松许可证，但资产为 CC BY-NC 4.0。 | 适合作为研究参考；NC 条款不适合通用再分发资产包。除非单个资产条款获确认，否则只作参考。 |

此表记录的是上游公开声明和集成范围，并不是对每个嵌套文件的法律认定。正式发布前仍需逐文件
检查许可证和声明。

## 建议的导入路径

第一批实际导入的是 robosuite v1.5.2 的 bottle、can 和 lemon。运行
`.venv/bin/python scripts/setup/download_scene_object.py`；脚本固定上游 commit，为每个文件校验
SHA256，下载 MIT 声明，并在被忽略的 `assets/objects/robosuite/<object>/v1/` 下生成小型物体片段（`object` 为
`bottle`、`can` 或 `lemon`）。
它不会安装整个 robosuite。
生成的物体片段可以用 `.venv_scene/bin/python scripts/dev/validate_scene_object.py` 独立编译检查。
如需将物体组合进现有 G1/Dex3 桌面场景，运行
`.venv_scene/bin/python scripts/setup/build_scene_with_object.py <object>`。
它会在被忽略的 `outputs/scenes/robosuite_<object>_43dof.xml` 写入生成场景，再通过现有启动器的
`--scene-xml` 参数运行。

第一批只导入一个简单 primitive 或一个许可证清晰的 robosuite 物体，并转换成 Teleopit 物体片段：

1. 可选视觉 mesh 和简化的原生碰撞 geom；
2. 明确的质量、摩擦、求解器参数和米制缩放；
3. 稳定的 body、geom 和 free-joint 名称；
4. 包含上游 URL、commit/tag、SPDX 状态和 SHA256 的清单条目；
5. 检查 XML 加载、接触和 reset 行为的 MuJoCo smoke test。

随后场景 XML 可以组合该片段，同时保持所需的 43 个执行器名称。这样资产导入与 PICO/XRoboToolkit
输入解耦，也不会把整个仿真框架复制进 Teleopit。

## 推荐工作顺序

1. 导入一个 robosuite primitive/物体，验证类似 cube 的抓取任务。
2. 增加与来源无关的物体适配器和物体清单条目。
3. 增加任务元数据和成功判定（`push`、`lift`、`place`）。
4. 将 RoboCasa 厨房资产放入单独的可选下载组进行评估。
