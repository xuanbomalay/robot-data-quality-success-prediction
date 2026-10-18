import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from robot_audit.audit import feature_columns, inspect_episode, make_splits
from robot_audit.features import EXPECTED_DIMS, arrays, content_hash, prefix_features
from robot_audit.modeling import choose_failure_threshold
from robot_audit.paths import Workspace
from robot_audit.policy import select_policy_conditions


def sample_frame(n=60):
    rng = np.random.default_rng(10)
    frame = pd.DataFrame({key: list(rng.normal(size=(n, dim))) for key, dim in EXPECTED_DIMS.items()})
    frame['timestamp'] = np.arange(n)/15
    frame['frame_index'] = np.arange(n)
    return frame


def test_future_frames_cannot_change_prefix_features():
    values = arrays(sample_frame())
    original = prefix_features(values)
    for key in EXPECTED_DIMS:
        values[key][30:] = 1e8
    assert prefix_features(values) == original


def test_source_labels_and_duration_not_model_features():
    values = prefix_features(arrays(sample_frame()))
    frame = pd.DataFrame([{**values, 'source': 'failure', 'success': 0, 'task': '', 'episode_length': 60}])
    selected = feature_columns(frame)
    assert set(selected) == set(values)
    assert not {'source', 'success', 'task', 'episode_length'} & set(selected)


def test_group_split_has_no_collector_overlap():
    rows = []
    for group in range(35):
        for j in range(12):
            k = f'{group}-{j}'
            rows.append({'collector': str(group), 'success': j % 2, 'episode_id': k, 'full_hash': k, 'prefix_hash': k})
    result = make_splits(pd.DataFrame(rows), 42)
    assert result.groupby('collector')['split'].nunique().max() == 1
    assert set(result['split']) == {'train', 'validation', 'test'}


def test_duplicate_content_cross_group_is_rejected():
    rows = [{'collector': str(i), 'success': j%2, 'episode_id': f'{i}-{j}',
             'full_hash': 'duplicate' if j == 0 else f'{i}-{j}', 'prefix_hash': f'{i}-{j}'} for i in range(21) for j in range(10)]
    with pytest.raises(ValueError, match='leakage'):
        make_splits(pd.DataFrame(rows), 42)


def test_injected_time_faults_are_detected():
    frame = sample_frame()
    frame.loc[12, 'timestamp'] = frame.loc[11, 'timestamp']
    frame.loc[20, 'frame_index'] = 19
    meta = pd.Series({'episode_id': 'test', 'collector': 'g', 'success': 0, 'length': 60, 'fps': 15, 'source': 'fixture'})
    record, _ = inspect_episode(meta, frame, 30)
    assert record['nonmonotonic_timestamp'] and record['frame_index_gap']


def test_bad_dimensions_do_not_silently_pass():
    frame = sample_frame()
    frame.at[0, 'action.joint_velocity'] = np.ones(6)
    meta = pd.Series({'episode_id': 'test', 'collector': 'g', 'success': 0, 'length': 60, 'fps': 15, 'source': 'fixture'})
    record, features = inspect_episode(meta, frame, 30)
    assert record['dimension_error'] and features is None


def test_hash_detects_copy_and_changed_motion():
    first = arrays(sample_frame())
    assert content_hash(first) == content_hash({k: v.copy() for k, v in first.items()})
    second = {k: v.copy() for k, v in first.items()}
    second['action.joint_velocity'][0, 0] += .1
    assert content_hash(first) != content_hash(second)


def test_scalar_gripper_storage_matches_declared_shape_one():
    frame = sample_frame()
    for key in ['observation.gripper_position', 'action.gripper_position']:
        frame[key] = frame[key].map(lambda a: float(a[0]))
    values = arrays(frame)
    assert values['action.gripper_position'].shape == (60, 1)
    assert len(prefix_features(values)) > 0


def test_path_traversal_and_outside_drive_rejected(tmp_path):
    if os.name == 'nt':
        with pytest.raises(ValueError):
            Workspace('C:/robot-audit-test')
    ws = Workspace(Path.cwd())
    with pytest.raises(ValueError, match='escapes'):
        ws.path('../outside.txt')


def test_drive_directory_without_mount_cannot_hold_checkpoints(monkeypatch):
    from pathlib import PurePosixPath
    from robot_audit.paths import require_cloud_mount

    cloud_root = PurePosixPath('/content/drive/MyDrive/robot-data-quality-success-prediction')
    monkeypatch.setattr(Path, 'is_mount', lambda self: False)
    with pytest.raises(RuntimeError, match='not mounted'):
        require_cloud_mount(cloud_root)
    monkeypatch.setattr(Path, 'is_mount', lambda self: True)
    require_cloud_mount(cloud_root)


def test_policy_filter_has_equal_retention_and_preserves_inputs():
    rng = np.random.default_rng(4)
    actions = {f'demo_{i}': rng.uniform(-.5, .5, (80, 7)).astype(np.float32) for i in range(100)}
    original = {k: v.copy() for k, v in actions.items()}
    conditions, truth, report = select_policy_conditions(actions)
    assert len(truth) == 10
    assert len(conditions['random_drop']) == len(conditions['audit_drop']) == len(conditions['oracle_drop']) == 90
    assert all(np.array_equal(actions[k], original[k]) for k in actions)
    assert all(np.isfinite(v).all() for v in conditions['contaminated'].values())


def test_audit_selection_does_not_read_injection_answer_identities(monkeypatch):
    from robot_audit import policy

    rng = np.random.default_rng(9)
    actions = {f'demo_{i}': rng.uniform(-.5, .5, (80, 7)).astype(np.float32) for i in range(100)}
    conditions, truth, original = policy.select_policy_conditions(actions)
    original_ids = {row['episode'] for row in truth}
    false_ids = [key for key in sorted(actions) if key not in original_ids][:len(truth)]
    false_answers = [{**row, 'episode': key} for row, key in zip(truth, false_ids)]
    # Hold detector inputs fixed while replacing the separate oracle answers.
    monkeypatch.setattr(policy, 'contaminate', lambda *args, **kwargs: (conditions['contaminated'], false_answers))
    _, _, changed = policy.select_policy_conditions(actions)
    assert changed['dropped']['audit_drop'] == original['dropped']['audit_drop']
    assert changed['dropped']['oracle_drop'] != original['dropped']['oracle_drop']


def test_threshold_uses_only_supplied_validation_values():
    y = np.array([1, 1, 0, 0])
    t = choose_failure_threshold(y, np.array([.1, .2, .7, .8]))
    assert .2 < t <= .7


def test_resumed_pilot_cannot_freeze_an_earlier_training_budget(tmp_path, monkeypatch):
    import json
    import sys
    from types import SimpleNamespace
    from robot_audit import policy

    ws = Workspace(tmp_path)
    checkpoint = ws.path('artifacts/policy/clean_seed5.pt')
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'synthetic checkpoint; torch loading is mocked')
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(load=lambda *a, **kw: {'step': 60000}))
    monkeypatch.setattr(policy, '_load', lambda _: {})
    requested_budgets = []

    def training(*args):
        requested_budgets.append(args[-1])
        return checkpoint

    monkeypatch.setattr(policy, 'train_bc', training)
    monkeypatch.setattr(policy, 'evaluate', lambda *args: [{'success': i < 16} for i in range(20)])
    config = {'policy': {'steps': 30000, 'pilot_seeds_start': 9000, 'pilot_episodes': 20, 'horizon': 400}}
    policy.pilot(ws, config)
    frozen = json.loads(ws.path('artifacts/policy/frozen_protocol.json').read_text())
    assert requested_budgets == [60000]
    assert frozen['steps'] == 60000


def test_policy_summary_rejects_a_training_budget_different_from_the_frozen_protocol(tmp_path):
    from robot_audit import policy

    ws = Workspace(tmp_path)
    ws.path('artifacts/policy').mkdir(parents=True)
    for condition in policy.CONDITIONS:
        for seed in [17, 29, 43]:
            ws.json(f'artifacts/policy/{condition}_seed{seed}_steps60000_eval_10000.json',
                    [{'condition': condition, 'training_seed': seed, 'training_steps': 60000,
                      'eval_seed': 10000, 'success': True, 'initial_state_sha256': 'synthetic'}])
    ws.json('artifacts/policy/frozen_protocol.json', {'steps': 30000, 'eval_seeds_start': 10000,
            'eval_episodes': 1, 'horizon': 400})
    with pytest.raises(RuntimeError, match='frozen training budget'):
        policy.summarize(ws, {'policy': {'train_seeds': [17, 29, 43], 'eval_episodes': 1}, 'seed': 42})
