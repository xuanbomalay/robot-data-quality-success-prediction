"""Recompute the published policy results from the 1,500-row CSV, without a simulator."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'src'))
from robot_audit.paths import Workspace

Workspace(root).setup()

import numpy as np
import pandas as pd
import yaml

reports = root / 'reports'


def read(name):
    return json.loads((reports / name).read_text(encoding='utf-8'))


config = yaml.safe_load((root / 'configs/project.yaml').read_text(encoding='utf-8'))
protocol = read('policy_protocol.json')
summary = read('policy_results.json')
preparation = read('policy_preparation.json')
receipt = read('policy_cloud_verification.json')
frame = pd.read_csv(reports / 'policy_rollouts.csv')
names = ['clean', 'contaminated', 'random_drop', 'audit_drop', 'oracle_drop']
seeds = [17, 29, 43]
starts = list(range(10000, 10100))
assert protocol['condition_order'] == names and protocol['train_seeds'] == seeds
assert protocol['eval_seeds_start'] == 10000 and protocol['eval_episodes'] == 100
assert protocol['steps'] == 60000 and protocol['pilot_passed'] is True
assert summary['status'] == 'complete' and summary['rollouts'] == len(frame) == 1500
assert summary['training_steps'] == 60000 and summary['evaluation_horizon'] == protocol['horizon'] == 400
assert not frame.isna().any().any()
assert frame.success.dtype == bool
assert set(frame.training_steps) == {protocol['steps']}
assert frame.steps.between(1, protocol['horizon']).all()
assert frame.initial_state_sha256.map(lambda value: bool(re.fullmatch('[0-9a-f]{64}', value))).all()
assert not frame.duplicated(['condition', 'training_seed', 'eval_seed']).any()
assert set(zip(frame.condition, frame.training_seed)) == {(c, s) for c in names for s in seeds}
for _, group in frame.groupby(['condition', 'training_seed']):
    assert len(group) == 100 and sorted(group.eval_seed) == starts
assert (frame.groupby('eval_seed').initial_state_sha256.nunique() == 1).all()
assert summary['paired_initial_states'] == 'PASS'
assert receipt['cloud_csv_matches_per_policy_records'] == 'PASS'
assert receipt['formal_raw_rows'] == 1500 and receipt['formal_raw_files'] == 15
assert hashlib.sha256((reports / 'policy_rollouts.csv').read_bytes()).hexdigest() == receipt['csv_sha256']

# Independent array implementation of the crossed-cluster calculation in policy.py.
cube = np.array([[[float(frame.loc[(frame.condition == c) & (frame.training_seed == s)
                                  & (frame.eval_seed == e), 'success'].item())
                   for c in names] for e in starts] for s in seeds])
rng = np.random.default_rng(config['seed'])
samples = np.empty((2000, len(names)))
for index in range(len(samples)):
    state_indices = rng.choice(len(starts), len(starts), replace=True)
    seed_indices = rng.choice(len(seeds), len(seeds), replace=True)
    samples[index] = cube[seed_indices][:, state_indices].mean(axis=(0, 1))
for index, name in enumerate(names):
    record = summary['conditions'][name]
    assert np.isclose(record['success_rate'], cube[:, :, index].mean(), atol=1e-12)
    np.testing.assert_allclose(record['ci95'], np.quantile(samples[:, index], [.025, .975]), atol=1e-12)
    for seed_index, seed in enumerate(seeds):
        saved = [r['success'] for r in summary['per_seed'] if r['condition'] == name and r['training_seed'] == seed]
        assert len(saved) == 1 and np.isclose(saved[0], cube[seed_index, :, index].mean())
difference = 100 * (cube[:, :, 3] - cube[:, :, 1]).mean()
assert np.isclose(difference, summary['audit_minus_contaminated_percentage_points'])
np.testing.assert_allclose(summary['audit_minus_contaminated_ci95_pp'],
                           np.quantile(100 * (samples[:, 3] - samples[:, 1]), [.025, .975]), atol=1e-12)
assert np.isclose(100 * (cube[:, :, 3] - cube[:, :, 2]).mean(),
                  summary['audit_minus_random_percentage_points'])

pilot = read('policy_pilot_rollouts.json')
assert len(pilot) == 20 and {r['eval_seed'] for r in pilot} == set(range(9000, 9020))
assert {r['training_steps'] for r in pilot} == {60000}
assert {r['training_seed'] for r in pilot} == {5} and {r['condition'] for r in pilot} == {'clean'}
assert sum(r['success'] for r in pilot) == 14
assert read('policy_pilot.json')['success_rate'] == .7
truth = read('policy_injection_truth.json')
assert len(truth) == len({r['episode'] for r in truth}) == preparation['injected_episodes'] == 18
assert sorted(Counter(r['kind'] for r in truth).values()) == [9, 9]
assert all(r['original_action_sha256'] != r['corrupted_action_sha256'] for r in truth)
injected = {r['episode'] for r in truth}
assert set(preparation['dropped']['oracle_drop']) == injected
assert len(set(preparation['dropped']['audit_drop']) & injected) == 10
assert len(set(preparation['dropped']['random_drop']) & injected) == 1
assert preparation['retained_per_condition'] == dict(zip(names, [180, 180, 162, 162, 162]))
assert preparation['data']['revision'] == protocol['revision'] == config['policy']['revision']
source = read('cloud_source_manifest.json')
assert source['torch'] == '2.11.0+cpu' and source['device'] == 'cpu'
for filename, digest in source['source_sha256'].items():
    normalized = (root / filename).read_text(encoding='utf-8').replace('\r\n', '\n').encode()
    assert hashlib.sha256(normalized).hexdigest() == digest, f'Cloud source mismatch: {filename}'
colab = read('colab_verification.json')
assert colab['status'] == 'PASS' and colab['mode'] == 'clean_kernel_sequential_resume'
assert colab['core_execution_counts'] == [1, 2, 3, 4, 5] and colab['error_outputs'] == 0
assert colab['formal_csv_sha256'] == receipt['csv_sha256']
qualitative = read('policy_qualitative.json')
assert len(qualitative['clips']) == 2
assert {clip['rollout']['condition'] for clip in qualitative['clips']} == {'contaminated', 'audit_drop'}
assert {clip['rollout']['eval_seed'] for clip in qualitative['clips']} == {20000}
assert len({clip['rollout']['initial_state_sha256'] for clip in qualitative['clips']}) == 1
showcase = qualitative.get('showcase')
assert showcase and showcase['rollout']['success'] is True
reference = frame[(frame.condition == 'clean') & (frame.training_seed == 17)
                  & frame.success & frame.eval_seed.gt(10000)].sort_values('eval_seed').iloc[0].to_dict()
assert showcase['rollout'] == reference
for clip in qualitative['clips'] + [showcase]:
    path = root / 'reports/figures' / clip['file']
    assert path.parent == root / 'reports/figures'
    assert path.stat().st_size == clip['bytes']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == clip['sha256']
print('POLICY ARTIFACT VERIFICATION PASS: 1,500 source-matched rows; budgets, paired starts, pilot, injection records, means and 2,000-resample intervals')
