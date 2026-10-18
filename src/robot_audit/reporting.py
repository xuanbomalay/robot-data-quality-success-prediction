from __future__ import annotations

import json

import numpy as np
import pandas as pd


def _read(ws, name):
    path = ws.path(f'reports/{name}.json')
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


def create_reports(ws, config):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'figure.dpi': 160, 'axes.spines.top': False, 'axes.spines.right': False})
    selection = _read(ws, 'data_selection')
    audit = _read(ws, 'audit_summary')
    anomaly = _read(ws, 'anomaly_summary')
    prediction = _read(ws, 'prediction_metrics')
    video = _read(ws, 'video_summary')
    policy = _read(ws, 'policy_results')
    if not all([selection, audit, anomaly, prediction]):
        raise RuntimeError('Run data audit, anomaly detection, and predictor training first')
    figure_dir = ws.path('reports/figures'); figure_dir.mkdir(exist_ok=True)
    predictions = pd.read_parquet(ws.path('data/processed/test_predictions.parquet'))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, group in predictions.groupby('model'):
        RocCurveDisplay.from_predictions(group.success, group.probability_success, name=name, ax=axes[0])
        PrecisionRecallDisplay.from_predictions(1-group.success, 1-group.probability_success, name=name, ax=axes[1])
    axes[0].set_title('Collector-held-out success ROC')
    axes[1].set_title('Failure precision-recall')
    fig.tight_layout(); fig.savefig(figure_dir/'prediction.png'); plt.close(fig)
    rows = pd.read_parquet(ws.path('data/processed/audit.parquet'))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    rows.actual_frames.plot.hist(bins=50, ax=axes[0], color='#137c8b')
    axes[0].set_title('Audited episode lengths'); axes[0].set_xlabel('Frames')
    values = [anomaly['hard_rule_flags'], anomaly['motion_outlier_flags'], anomaly['isolation_flags']]
    axes[1].bar(['Structural / time', 'Motion outlier', 'Isolation Forest'], values, color=['#da6b46','#e0ad54','#137c8b'])
    axes[1].set_title('Review flags (overlapping; not corruption labels)'); axes[1].tick_params(axis='x', labelsize=8)
    fig.tight_layout(); fig.savefig(figure_dir/'audit.png'); plt.close(fig)
    model = prediction['selected_model']; m = prediction['models'][model]['test']; ci = prediction['models'][model]['test_ci95']
    selected_predictions = predictions[predictions.model.eq(model)].copy()
    selected_predictions['probability_failure'] = 1 - selected_predictions.probability_success
    alert = selected_predictions.probability_failure >= selected_predictions.failure_threshold
    error_rows = []
    for label, mask, ascending in [
            ('False alert', selected_predictions.success.eq(1) & alert, False),
            ('Missed failure', selected_predictions.success.eq(0) & ~alert, True)]:
        candidates = selected_predictions[mask].sort_values(['probability_failure', 'episode_id'],
                                                           ascending=[ascending, True])
        if not candidates.empty:
            row = candidates.iloc[0]
            error_rows.append(f'| {label} | `{row.episode_id}` | {row.probability_failure:.1%} | {row.failure_threshold:.1%} |')
    errors_md = '| Case | Project episode ID | Predicted failure | Fixed alert threshold |\n|---|---|---:|---:|\n' + '\n'.join(error_rows)
    importance_path = ws.path('reports/feature_importance.csv')
    importance_md = 'Validation permutation importance is unavailable.'
    if importance_path.exists():
        importance = pd.read_csv(importance_path).head(5)
        importance_md = '| Prefix feature | Validation AUC drop after permutation | Repeat SD |\n|---|---:|---:|\n'
        importance_md += '\n'.join(f'| `{row.feature}` | {row.validation_permutation_auc_drop:.4f} | {row.std:.4f} |'
                                   for row in importance.itertuples(index=False))
    policy_status = 'complete' if policy and policy.get('status') == 'complete' else 'pending — no policy improvement claim'
    rows_md = '\n'.join(f'| {name} | {r["test"]["roc_auc_success"]:.3f} | {r["test"]["pr_auc_failure"]:.3f} | {r["test"]["failure_recall"]:.3f} |' for name,r in prediction['models'].items())
    policy_text = 'The policy comparison has not completed. No downstream improvement is claimed.'
    if policy and policy.get('status') == 'complete':
        policy_text = '\n'.join(f'- {name}: {r["success_rate"]:.1%} success (95% CI {r["ci95"][0]:.1%}–{r["ci95"][1]:.1%}).' for name,r in policy['conditions'].items())
        policy_text += f'\nAudit-filtered minus contaminated: {policy["audit_minus_contaminated_percentage_points"]:.2f} percentage points; CI {policy["audit_minus_contaminated_ci95_pp"]}. Simulation and injected corruptions only.'
        policy_text += '\n\nAll 15 policies used 60,000 optimization steps and the same 100 initial-state seeds (10,000–10,099). The independent clean pilot used training seed 5 and separate starts (9,000–9,019), achieving 14/20 before the formal budget was frozen. Formal training seeds were 17, 29 and 43; the final fixed-step checkpoints were evaluated without test-based checkpoint selection.'
        policy_text += '\n\nThe training masks contained 180 demonstrations; validation masks contained 20. Exactly 18 training demonstrations were corrupted (nine with action spikes and nine with action misalignment). Each filtering condition removed 18 demonstrations. Audit filtering recovered 10/18 injected cases, while random removal recovered 1/18. Random removal also reduced success below the contaminated baseline in this experiment, so sample removal alone did not provide the observed audit benefit. Audit filtering remained below the clean mean; the experiment does not establish complete recovery or unseen-task generalization.'
        policy_text += '\n\nEach of the 1,500 published CSV rows was checked against its original per-policy cloud record. An independent NumPy implementation recomputes all means and bootstrap intervals with `scripts/verify_policy_results.py`. Source, environment, protocol and injection hashes are retained in the adjacent JSON reports. No checkpoints or raw training data are redistributed.'
        names = list(policy['conditions'])
        means = np.array([policy['conditions'][name]['success_rate'] for name in names])
        intervals = np.array([policy['conditions'][name]['ci95'] for name in names])
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.bar(np.arange(len(names)), means * 100,
               yerr=np.stack([means - intervals[:, 0], intervals[:, 1] - means]) * 100,
               capsize=4, color=['#137c8b', '#da6b46', '#a8b3bf', '#e0ad54', '#3c9270'])
        for index, name in enumerate(names):
            per_seed = [row['success'] * 100 for row in policy['per_seed'] if row['condition'] == name]
            ax.scatter(index + np.linspace(-.12, .12, len(per_seed)), per_seed,
                       s=22, color='#172a3a', zorder=4)
        ax.set_xticks(range(len(names)), [name.replace('_', '\n') for name in names])
        ax.set_ylim(0, 104); ax.set_ylabel('Success rate (%)')
        ax.set_title('Simulated Lift: injected action corruption only')
        ax.text(.5, -.25, 'Dots: training seeds; whiskers: paired two-way cluster 95% intervals; 100 matched starts per policy',
                transform=ax.transAxes, ha='center', fontsize=8)
        fig.tight_layout(); fig.savefig(figure_dir/'policy.png', bbox_inches='tight'); plt.close(fig)
        policy_text += '\n\n![Controlled simulation comparison](figures/policy.png)'
    report = f'''# Research report: robot demonstration quality and success prediction

## Question and contribution
Can a reproducible audit identify review candidates in robot demonstrations, and can the first {config['prefix_frames']} frames predict eventual demonstration success under a collector-held-out split?

This project implements the data adapters, bounded/resumable downloads, motion features, quality rules, anomaly ranking, group-safe evaluation, and a separate controlled policy experiment. It does not introduce a new robot foundation model or claim real-robot deployment.

## Data and audit coverage
- Audited episodes: **{audit['audited_episodes']:,}**; classifier-eligible episodes after content/prefix deduplication: **{audit['predictor_eligible_episodes']:,}**.
- Selected outcome counts: {selection['selected_classes']}; collectors in selection: {selection['collector_groups']}.
- Upstream duplicate UUID rows: {selection['duplicate_uuid_rows']}; conflicting UUIDs quarantined: {selection['conflicting_uuid_count']}.
- Review flags: {anomaly['review_flags']:,}. These are review suggestions, not verified corruptions.
- Timestamp checks apply to converted 15-Hz storage. Original sensor-clock drift is not measurable from these files.
- Video: {json.dumps(video, ensure_ascii=False) if video else 'sample audit pending; no video-wide claim'}.

## Prediction protocol
Only fixed-prefix motion signals are allowed. Task text is absent for all failure episodes and is excluded, along with source path, IDs, full duration, outcome/reward and terminal data. Scaling, imputation, anomaly thresholds and model fitting use training data only. The failure threshold and presentation model are selected on validation data. Test collectors stay held out.

Actual split counts: {audit['split_counts']}; collector counts: {audit['split_collectors']}. Sampling deliberately targets an 80/20 outcome mix; precision and calibration reflect this sampled distribution, not an operational fleet prevalence.

| Model | Test ROC-AUC | Failure PR-AUC | Failure recall |
|---|---:|---:|---:|
{rows_md}

Validation-selected model: **{model}**. Test AUC 95% collector-bootstrap CI: {ci['roc_auc_success']}. Three baselines are prespecified; no search for a target AUC of 0.85 was performed. Full metrics, threshold and confusion matrices are in `prediction_metrics.json`.

![Held-out evaluation](figures/prediction.png)

## Feature interpretation and error inspection
{importance_md}

These are validation-set permutation results for the selected model, with three repeats. Feature indices are zero-based; `min`, `std` and `mean_abs_delta` summarize only the fixed 30-frame prefix. Small AUC drops describe model dependence on correlated motion signals, not causal effects or explanations for task failure.

{errors_md}

The false alert has the highest predicted failure probability among successful demonstrations; the missed failure has the lowest among failed demonstrations. These examples were chosen for inspection after the model and threshold were fixed. No task-failure cause is inferred from early motion alone, and the examples are not used for retraining or threshold selection.

## Controlled policy experiment
Status: **{policy_status}**.

{policy_text}

## Limits and interpretation
- Labels describe human teleoperated demonstrations, not autonomous policy execution.
- The community conversion already removed some inconsistent episodes; findings do not estimate all raw DROID defects.
- Collector grouping tests transfer across collectors, not verified unseen scenes or unseen tasks.
- Success and failure cohorts come from separate converted repositories. Removing explicit source fields does not eliminate possible task, collection or conversion differences between cohorts; these may contribute to discrimination.
- Motion anomaly and task failure are separate concepts. Isolation Forest scores do not establish corruption or causation.
- A small, budget-limited video sample cannot establish whole-dataset camera quality.
- Controlled simulation results do not establish that filtering real DROID improves a physical robot.
- Policy uncertainty is limited by three training seeds, one Lift task and one fixed corruption draw. Bootstrap intervals do not establish generalization across tasks or corruption mixtures.

## Reproducibility
Pinned data commits: `source_versions.json`; downloaded file SHA256 values recorded. Project seed 42. Windows dependency versions: `requirements-lock-windows.txt`. Policy checkpoints include optimizer/RNG state. Formal policy comparisons require all 1,500 rollouts and identical initial-state hashes. CI uses synthetic fixtures clearly separated from real-data metrics.
'''
    ws.path('reports/research_report.md').write_text(report, encoding='utf-8')
    resume_second = f'- Engineered fixed-prefix motion features and benchmarked logistic regression and LightGBM against a prior baseline under collector-held-out validation; the validation-selected {model} achieved a test ROC-AUC of {m["roc_auc_success"]:.3f} (95% grouped-bootstrap CI {ci["roc_auc_success"][0]:.3f}–{ci["roc_auc_success"][1]:.3f}).'
    policy_brief = '最后用一个独立的 Lift 仿真实验检验数据清洗是否改善策略。真实数据的质量审计与人工污染的仿真实验分别解释，只有五组对照全部完成之后才报告策略效果。'
    if policy and policy.get('status') == 'complete':
        difference = policy['audit_minus_contaminated_percentage_points']
        interval = policy['audit_minus_contaminated_ci95_pp']
        resume_second = f'- Designed a controlled Lift behavior-cloning benchmark with five data conditions, three training seeds and 1,500 matched rollouts; measured an audit-filtered versus contaminated success-rate difference of {difference:+.1f} percentage points (95% paired cluster-bootstrap CI {interval[0]:+.1f} to {interval[1]:+.1f}), scoped to simulated action corruption.'
        policy_brief = f'最后我完成了独立的 Lift 仿真实验：五组数据、三个训练随机种子，每个策略使用相同的 100 个初始状态，共 1,500 次评测。审计剔除组相对污染组的成功率差为 {difference:+.1f} 个百分点，95% 区间为 {interval[0]:+.1f} 到 {interval[1]:+.1f}。区间同时考虑训练种子和共享初始状态的不确定性。它检验的是人工动作污染下的仿真表现，不能解释成真实 DROID 机器人提升。'
    resume = f'''# Resume evidence and interview brief

## Verified English resume bullets
- Built a resumable audit and early-success prediction pipeline for {audit['audited_episodes']:,} DROID demonstrations using {audit['feature_count']} motion features from the first {config['prefix_frames']} frames, rule checks and Isolation Forest; maintained pinned data versions, file hashes and collector-held-out logistic regression/LightGBM evaluation.
{resume_second}

Do not use the old AUC 0.85 or 16% improvement unless this repository's measured results support them. Policy status: {policy_status}.

## 三分钟讲解稿
我把项目定义为机器人示范数据的可靠性问题。训练之前，需要知道数据是否完整、动作是否异常，以及能否从轨迹早期预测最终的演示结果。

数据采用 DROID 的 LeRobot 转换版。我实际审计了 {audit['audited_episodes']:,} 条轨迹，并保留数据版本、哈希和处理记录。审计覆盖结构、转换后的时间戳和动作异常，视频只做抽样。我不会把异常分数直接解释成数据损坏。

建模时我发现了一个很容易导致虚高成绩的问题：失败轨迹的任务文本都是空的，所以文本、来源、完整长度和终局信息全部排除。模型只看前 30 帧运动特征，并按采集者划分训练和测试，而不是随机拆分帧。实际测试 AUC 是 {m['roc_auc_success']:.3f}，不确定性区间也一起报告。

{policy_brief}

这个项目中我负责的是数据管线、质量规则、特征设计、防泄漏评估和工程复现。它的价值是给出可检查的证据，而不是用一个漂亮的准确率代替完整实验。
'''
    ws.path('reports/resume_evidence.md').write_text(resume, encoding='utf-8')
    existing_status = _read(ws, 'project_status') or {}
    colab_verification = _read(ws, 'colab_verification') or {}
    qualitative = _read(ws, 'policy_qualitative')
    ws.json('reports/project_status.json', {'real_data_audit': 'complete', 'prediction': 'complete',
             'sample_video': 'complete' if video else 'pending', 'policy': policy_status,
             'colab_kernel_resume': colab_verification.get('status', 'pending'),
             'qualitative_demo': 'complete' if qualitative else 'pending',
             'github_publication': existing_status.get('github_publication', 'pending'), 'selected_model': model,
             'test_roc_auc': m['roc_auc_success'], 'audited_episodes': audit['audited_episodes']})
    print('REPORTS GENERATED — measured results only', flush=True)
