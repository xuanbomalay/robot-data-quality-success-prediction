# Resume evidence and interview brief

## Verified English resume bullets
- Built a reproducible audit pipeline for 15,000 DROID robot demonstrations in LeRobot v3 format, checking trajectory structure, converted timestamps, action anomalies and sampled video alignment; maintained pinned data versions and file hashes.
- Engineered fixed-prefix motion features and benchmarked logistic regression and LightGBM against a prior baseline under collector-held-out validation; the validation-selected lightgbm achieved a test ROC-AUC of 0.592 (95% grouped-bootstrap CI 0.473–0.664).

Do not use the old AUC 0.85 or 16% improvement unless this repository's measured results support them. Policy status: pending — no policy improvement claim.

## 三分钟讲解稿
我把项目定义为机器人示范数据的可靠性问题。训练之前，需要知道数据是否完整、动作是否异常，以及能否从轨迹早期预测最终的演示结果。

数据采用 DROID 的 LeRobot 转换版。我实际审计了 15,000 条轨迹，并保留数据版本、哈希和处理记录。审计覆盖结构、转换后的时间戳和动作异常，视频只做抽样。我不会把异常分数直接解释成数据损坏。

建模时我发现了一个很容易导致虚高成绩的问题：失败轨迹的任务文本都是空的，所以文本、来源、完整长度和终局信息全部排除。模型只看前 30 帧运动特征，并按采集者划分训练和测试，而不是随机拆分帧。实际测试 AUC 是 0.592，不确定性区间也一起报告。

最后用一个独立的 Lift 仿真实验检验数据清洗是否改善策略。真实数据的质量审计与人工污染的仿真实验分别解释，只有五组对照全部完成之后才报告策略效果。

这个项目中我负责的是数据管线、质量规则、特征设计、防泄漏评估和工程复现。它的价值是给出可检查的证据，而不是用一个漂亮的准确率代替完整实验。
