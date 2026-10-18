from __future__ import annotations

import json
import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import StratifiedGroupKFold

from .data import episodes
from .features import arrays, content_hash, prefix_features, trajectory_summary


def inspect_episode(meta, frame, prefix: int) -> tuple[dict, dict | None]:
    record = {'episode_id': meta.episode_id, 'collector': meta.collector, 'success': int(meta.success),
              'expected_frames': int(meta.length), 'actual_frames': len(frame), 'source': meta.source,
              'dimension_error': False, 'error': ''}
    time = frame.timestamp.to_numpy(dtype=float)
    dt = np.diff(time)
    record.update({'frame_count_mismatch': len(frame) != meta.length,
                   'nonmonotonic_timestamp': bool(np.any(dt <= 0)),
                   'timestamp_nonfinite': bool(np.any(~np.isfinite(time))),
                   'converted_interval_max_error_s': float(np.max(np.abs(dt - 1 / float(meta.fps)))) if len(dt) else 0.0,
                   'frame_index_gap': bool(np.any(np.diff(frame.frame_index.to_numpy()) != 1)),
                   'video_status': 'not_checked'})
    try:
        values = arrays(frame)
        record.update(trajectory_summary(values))
        record['full_hash'] = content_hash(values)
        record['prefix_hash'] = content_hash(values, prefix)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            features = prefix_features(values, prefix)
        features.update({k: record[k] for k in ['episode_id', 'collector', 'success', 'full_hash', 'prefix_hash']})
    except (ValueError, TypeError) as exc:
        record['dimension_error'] = True
        record['error'] = str(exc)
        features = None
    return record, features


def make_splits(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    result = frame.copy().reset_index(drop=True)
    n = min(7, int(result.collector.nunique()))
    if n < 3:
        raise ValueError('At least three independent collector groups are required')
    splitter = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed)
    folds = list(splitter.split(result, result.success, groups=result.collector))
    result['split'] = 'train'
    result.loc[folds[0][1], 'split'] = 'test'
    result.loc[folds[1][1], 'split'] = 'validation'
    for split, subset in result.groupby('split'):
        if subset.success.nunique() < 2:
            raise ValueError(f'{split} lacks both outcome classes')
    for name in ['collector', 'full_hash', 'prefix_hash', 'episode_id']:
        if (result.groupby(name)['split'].nunique() > 1).any():
            raise ValueError(f'Cross-split leakage in {name}')
    return result


def audit(ws, config: dict) -> pd.DataFrame:
    from . import features as feature_module
    fingerprint = hashlib.sha256(Path(__file__).read_bytes() + Path(feature_module.__file__).read_bytes()
                                 + json.dumps({'prefix': config['prefix_frames'], 'sources': config['sources']}, sort_keys=True).encode()).hexdigest()
    cache = ws.path('artifacts/episode_audit.jsonl')
    cached = {}
    if cache.exists():
        for line in cache.read_text(encoding='utf-8').splitlines():
            try:
                item = json.loads(line)
                if item.get('fingerprint') == fingerprint:
                    cached[item['audit']['episode_id']] = item
            except json.JSONDecodeError:
                continue
    audit_rows, feature_rows = [], []
    with cache.open('a', encoding='utf-8') as handle:
        for i, (meta, frame) in enumerate(episodes(ws, config)):
            if meta.episode_id in cached:
                item = cached[meta.episode_id]
                record, features = item['audit'], item['features']
            else:
                record, features = inspect_episode(meta, frame, config['prefix_frames'])
                # NaNs are allowed in internal JSONL only and are imputed on training data later.
                handle.write(json.dumps({'fingerprint': fingerprint, 'audit': record, 'features': features}, ensure_ascii=False) + '\n')
                handle.flush()
            audit_rows.append(record)
            if features is not None:
                feature_rows.append(features)
            if (i + 1) % 500 == 0:
                print(f'AUDIT {i+1} episodes', flush=True)
    raw_audit = pd.DataFrame(audit_rows)
    features = pd.DataFrame(feature_rows)
    if features.empty:
        raise RuntimeError('No episodes have a valid feature schema; inspect audit JSONL errors before continuing.')
    # Drop every member of a duplicate content/prefix cluster, conservatively including conflicts.
    duplicate = features.duplicated('full_hash', keep=False) | features.duplicated('prefix_hash', keep=False)
    duplicate_ids = set(features.loc[duplicate, 'episode_id'])
    eligible = features.loc[~duplicate].copy()
    raw_audit['duplicate_content_or_prefix'] = raw_audit.episode_id.isin(duplicate_ids)
    eligible = make_splits(eligible, config['seed'])
    eligible.to_parquet(ws.path('data/processed/features.parquet'), index=False)
    raw_audit = raw_audit.merge(eligible[['episode_id', 'split']], on='episode_id', how='left')
    raw_audit['split'] = raw_audit['split'].fillna('excluded')
    raw_audit.to_parquet(ws.path('data/processed/audit.parquet'), index=False)
    summary = {'audited_episodes': len(raw_audit), 'predictor_eligible_episodes': len(eligible),
               'duplicate_content_or_prefix_episodes_excluded': int(duplicate.sum()),
               'structural_errors': int(raw_audit.dimension_error.sum()),
               'rule_counts': {k: int(raw_audit[k].sum()) for k in ['frame_count_mismatch', 'nonmonotonic_timestamp', 'timestamp_nonfinite', 'frame_index_gap']},
               'timestamp_semantics': '15-Hz converted timestamps; original sensor-clock drift not measurable',
               'split_counts': eligible['split'].value_counts().to_dict(),
               'split_classes': {s: g.success.value_counts().to_dict() for s, g in eligible.groupby('split')},
               'split_collectors': eligible.groupby('split').collector.nunique().to_dict(),
               'feature_count': len(feature_columns(eligible)), 'prefix_frames': config['prefix_frames'],
               'video_coverage': 'separate sample; never all episodes', 'leakage_checks': 'PASS'}
    ws.json('reports/audit_summary.json', summary)
    return raw_audit


def feature_columns(frame: pd.DataFrame) -> list[str]:
    allowed_prefixes = ('observation_joint_', 'observation_gripper_', 'observation_cartesian_', 'action_joint_', 'action_gripper_')
    cols = [c for c in frame if c.startswith(allowed_prefixes)]
    if not cols:
        raise ValueError('No allow-listed motion features')
    forbidden = ('success', 'source', 'task', 'timestamp', 'length', 'hash', 'episode', 'collector')
    if any(any(term in c for term in forbidden) for c in cols):
        raise ValueError('Forbidden model feature')
    return cols


def detect_anomalies(ws, config: dict) -> pd.DataFrame:
    frame = pd.read_parquet(ws.path('data/processed/audit.parquet'))
    columns = ['nonfinite_fraction', 'frames', 'action_max_abs', 'action_delta_p95', 'action_delta_max',
               'joint_delta_p95', 'joint_delta_max', 'position_delta_p95', 'position_delta_max', 'position_path_length']
    x = frame[columns].replace([np.inf, -np.inf], np.nan)
    train = frame.split.eq('train')
    medians = x.loc[train].median()
    x = x.fillna(medians).fillna(0)
    x = np.log1p(x.clip(lower=0))
    detector = IsolationForest(n_estimators=300, max_samples=min(1024, int(train.sum())),
                               contamination=.1, random_state=config['seed'], n_jobs=4)
    detector.fit(x.loc[train])
    score = -detector.score_samples(x)
    threshold = float(np.quantile(score[train], .9))
    frame['isolation_score'] = score
    frame['isolation_flag'] = score >= threshold
    baseline = x.loc[train].median()
    mad = (x.loc[train] - baseline).abs().median().clip(lower=.05)
    robust = ((x - baseline).abs() / (1.4826 * mad)).max(axis=1)
    frame['motion_outlier_flag'] = robust > 12
    frame['hard_rule_flag'] = (frame.dimension_error | frame.frame_count_mismatch | frame.nonmonotonic_timestamp
                              | frame.timestamp_nonfinite | frame.frame_index_gap | frame.nonfinite_fraction.fillna(1).gt(0)
                              | frame.converted_interval_max_error_s.gt(.001))
    frame['review_flag'] = frame.hard_rule_flag | frame.motion_outlier_flag | frame.isolation_flag | frame.duplicate_content_or_prefix
    frame['reasons'] = frame.apply(lambda r: '; '.join(name for name, flag in [
        ('structural_or_time_rule', r.hard_rule_flag), ('motion_outlier', r.motion_outlier_flag),
        ('isolation_forest', r.isolation_flag), ('duplicate_content', r.duplicate_content_or_prefix)] if flag), axis=1)
    frame.to_parquet(ws.path('data/processed/audit.parquet'), index=False)
    ws.json('reports/anomaly_summary.json', {'episodes': len(frame), 'review_flags': int(frame.review_flag.sum()),
            'hard_rule_flags': int(frame.hard_rule_flag.sum()), 'motion_outlier_flags': int(frame.motion_outlier_flag.sum()),
            'isolation_flags': int(frame.isolation_flag.sum()), 'threshold_fit_split': 'train',
            'isolation_threshold': threshold, 'interpretation': 'Review candidates, not verified corruption; failures are not quality labels.'})
    return frame
