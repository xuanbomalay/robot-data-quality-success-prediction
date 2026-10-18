# Resume evidence and interview brief

## Verified English resume bullets
- Built a resumable audit and early-success prediction pipeline for 15,000 DROID demonstrations using 156 motion features from the first 30 frames, rule checks and Isolation Forest; maintained pinned data versions, file hashes and collector-held-out logistic regression/LightGBM evaluation.
- Designed a controlled Lift behavior-cloning benchmark with five data conditions, three training seeds and 1,500 matched rollouts; measured an audit-filtered versus contaminated success-rate difference of +35.0 percentage points (95% paired cluster-bootstrap CI +21.0 to +48.7), scoped to simulated action corruption.

Do not use the old AUC 0.85 or 16% improvement unless this repository's measured results support them. Policy status: complete.

## 三分钟讲解稿
我把项目定义为机器人示范数据的可靠性问题。训练之前，需要知道数据是否完整、动作是否异常，以及能否从轨迹早期预测最终的演示结果。

数据采用 DROID 的 LeRobot 转换版。我实际审计了 15,000 条轨迹，并保留数据版本、哈希和处理记录。审计覆盖结构、转换后的时间戳和动作异常，视频只做抽样。我不会把异常分数直接解释成数据损坏。

建模时我发现了一个很容易导致虚高成绩的问题：失败轨迹的任务文本都是空的，所以文本、来源、完整长度和终局信息全部排除。模型只看前 30 帧运动特征，并按采集者划分训练和测试，而不是随机拆分帧。实际测试 AUC 是 0.592，不确定性区间也一起报告。

最后我完成了独立的 Lift 仿真实验：五组数据、三个训练随机种子，每个策略使用相同的 100 个初始状态，共 1,500 次评测。审计剔除组相对污染组的成功率差为 +35.0 个百分点，95% 区间为 +21.0 到 +48.7。区间同时考虑训练种子和共享初始状态的不确定性。它检验的是人工动作污染下的仿真表现，不能解释成真实 DROID 机器人提升。

这个项目中我负责的是数据管线、质量规则、特征设计、防泄漏评估和工程复现。它的价值是给出可检查的证据，而不是用一个漂亮的准确率代替完整实验。
