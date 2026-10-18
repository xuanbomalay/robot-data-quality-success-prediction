from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .download import fetch_many, tree


def identity(value: str) -> str:
    return hashlib.sha256(value.encode('utf-8')).hexdigest()[:20]


def fetch_metadata(ws, config: dict) -> dict:
    output = {}
    for source, spec in config['sources'].items():
        entries = [x for x in tree(spec['repo'], spec['revision'], 'meta')
                   if x['path'].endswith(('.parquet', '.json'))]
        output[source] = fetch_many(ws, spec['repo'], spec['revision'], entries, workers=2)
    ws.json('reports/internal/metadata_manifest.json', output)
    return output


def select_episodes(ws, config: dict, small: bool = False) -> pd.DataFrame:
    frames = []
    for source, spec in config['sources'].items():
        folder = ws.path(Path('data/raw') / spec['repo'] / 'meta/episodes')
        frame = pd.concat([pd.read_parquet(p) for p in sorted(folder.rglob('*.parquet'))], ignore_index=True)
        frame['source'] = source
        frame['success'] = spec['label']
        frame['collector'] = frame.uuid.map(lambda s: identity('+'.join(s.split('+')[:2])))
        frame['episode_id'] = frame.uuid.map(identity)
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    counts = frame.groupby('uuid').success.nunique()
    conflicts = set(counts[counts > 1].index)
    duplicate_rows = int(frame.duplicated('uuid', keep='first').sum())
    summary = {'metadata_rows': len(frame), 'unique_uuid': int(frame.uuid.nunique()),
               'duplicate_uuid_rows': duplicate_rows, 'conflicting_uuid_count': len(conflicts),
               'shorter_than_prefix': int((frame.length < config['prefix_frames']).sum()),
               'blank_task_by_label': frame.assign(blank=frame.task.fillna('').eq('')).groupby('success').blank.sum().to_dict()}
    frame = frame[~frame.uuid.isin(conflicts)].drop_duplicates('uuid', keep='first')
    frame = frame[frame.length >= config['prefix_frames']]
    chosen = []
    for source in config['sources']:
        candidates = frame[frame.source == source].sort_values('uuid')
        n = min(150 if source == 'success' else 50, len(candidates)) if small else config['sample'][source]
        if len(candidates) < n:
            raise ValueError(f'Not enough eligible {source} episodes: {len(candidates)} < {n}')
        chosen.append(candidates.sample(n=n, random_state=config['seed']))
    selected = pd.concat(chosen, ignore_index=True)
    selected.to_parquet(ws.path('data/processed/selection.parquet'), index=False)
    summary['selected_episodes'] = len(selected)
    summary['selected_classes'] = selected.success.value_counts().to_dict()
    summary['collector_groups'] = int(selected.collector.nunique())
    summary['sampling'] = 'uniform within outcome stratum; source-derived outcomes; seed=42'
    ws.json('reports/data_selection.json', summary)
    print(f'SELECTION {len(selected)} episodes, {selected.collector.nunique()} collectors', flush=True)
    return selected


def fetch_controls(ws, config: dict) -> None:
    selected = pd.read_parquet(ws.path('data/processed/selection.parquet'))
    manifest = []
    for source, spec in config['sources'].items():
        subset = selected[selected.source == source]
        paths = {f'data/chunk-{int(row["data/chunk_index"]):03d}/file_{int(row["data/file_index"]):03d}.parquet'
                 for _, row in subset.iterrows()}
        entries = [x for x in tree(spec['repo'], spec['revision'], 'data') if x['path'] in paths]
        if {x['path'] for x in entries} != paths:
            raise RuntimeError('Selected control shards missing upstream')
        manifest.extend(fetch_many(ws, spec['repo'], spec['revision'], entries))
    ws.json('reports/internal/control_manifest.json', manifest)
    ws.json('reports/source_versions.json', {'sources': config['sources'],
            'control_shards': len(manifest), 'control_bytes': sum(x['size'] for x in manifest),
            'checksums': [{'source': x['repo'], 'file': x['path'], 'sha256': x['sha256']} for x in manifest]})


SIGNALS = ('observation.joint_position', 'observation.joint_velocity',
           'observation.gripper_position', 'observation.cartesian_position',
           'action.joint_velocity', 'action.gripper_position')


def episodes(ws, config: dict):
    selected = pd.read_parquet(ws.path('data/processed/selection.parquet'))
    for (source, chunk, file_id), group in selected.groupby(['source', 'data/chunk_index', 'data/file_index'], sort=True):
        spec = config['sources'][source]
        rel = f'data/chunk-{int(chunk):03d}/file_{int(file_id):03d}.parquet'
        frame = pd.read_parquet(ws.path(Path('data/raw') / spec['repo'] / rel), columns=['episode_index', 'frame_index', 'timestamp', *SIGNALS])
        grouped = frame.groupby('episode_index', sort=False)
        for _, meta in group.iterrows():
            try:
                data = grouped.get_group(meta.episode_index).sort_values('frame_index', kind='stable')
            except KeyError as exc:
                raise RuntimeError(f'Selected episode missing from shard: {meta.episode_id}') from exc
            yield meta, data
