from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .download import fetch_many, tree


def audit_videos(ws, config, episode_count=30):
    import av
    selection = pd.read_parquet(ws.path('data/processed/selection.parquet'))
    audit = pd.read_parquet(ws.path('data/processed/audit.parquet'))
    selection = selection.merge(audit[['episode_id', 'review_flag']], on='episode_id', validate='one_to_one')
    # Balanced flagged/unflagged, success/failure selection; sample is explicitly nonrepresentative.
    candidates = []
    for _, group in selection.groupby(['success', 'review_flag'], dropna=False):
        candidates.append(group.sample(n=min(len(group), (episode_count+3)//4), random_state=config['seed']))
    candidates = pd.concat(candidates).head(episode_count)
    budget = int(config['video_quota_gb']*1024**3)
    plan, chosen, sizes = {}, [], {}
    camera = 'observation.images.left_external'
    for source, spec in config['sources'].items():
        entries = tree(spec['repo'], spec['revision'], f'videos/{camera}')
        sizes[source] = {r['path']: r for r in entries}
        plan[source] = {}
    consumed = 0
    for _, row in candidates.iterrows():
        source = row.source
        rel = f'videos/{camera}/chunk-{int(row[f"videos/{camera}/chunk_index"]):03d}/file_{int(row[f"videos/{camera}/file_index"]):03d}.mp4'
        entry = sizes[source].get(rel)
        if entry is None:
            chosen.append((row, rel, 'missing_upstream')); continue
        cost = 0 if rel in plan[source] else int(entry['size'])
        if consumed + cost > budget:
            continue
        consumed += cost
        plan[source][rel] = entry
        chosen.append((row, rel, 'download'))
    for source, entries in plan.items():
        spec = config['sources'][source]
        fetch_many(ws, spec['repo'], spec['revision'], list(entries.values()), workers=2)
    records = []
    frame_dir = ws.path('artifacts/video_frames'); frame_dir.mkdir(exist_ok=True)
    for row, rel, status in chosen:
        record = {'episode_id': row.episode_id, 'camera': camera, 'success': int(row.success),
                  'review_flag': bool(row.review_flag), 'status': status, 'expected_frames': int(row.length)}
        if status == 'missing_upstream':
            records.append(record); continue
        path = ws.path(Path('data/raw') / config['sources'][row.source]['repo'] / rel)
        start = float(row[f'videos/{camera}/from_timestamp'])
        end = float(row[f'videos/{camera}/to_timestamp'])
        times = []
        try:
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                container.seek(int(max(0, start-.5)/float(stream.time_base)), stream=stream, backward=True)
                for frame in container.decode(stream):
                    if frame.pts is None:
                        continue
                    t = float(frame.pts * stream.time_base)
                    if t < start - .25/float(row.fps):
                        continue
                    if t >= end - .25/float(row.fps):
                        break
                    if not times:
                        frame.to_image().resize((640, 360)).save(frame_dir / f'{row.episode_id}.jpg')
                    times.append(t)
            record.update(status='decoded', decoded_frames=len(times),
                          frame_count_match=len(times) == int(row.length),
                          pts_monotonic=bool(np.all(np.diff(times) > 0)),
                          converted_start_error_s=abs(times[0]-start) if times else None,
                          converted_max_interval_error_s=float(np.max(np.abs(np.diff(times)-1/float(row.fps)))) if len(times)>1 else None)
        except Exception as exc:
            record.update(status='decode_error', error=f'{type(exc).__name__}: {exc}')
        records.append(record)
        print(f'VIDEO {len(records)}/{len(chosen)} {record["status"]}', flush=True)
    ws.json('reports/internal/video_audit_records.json', records)
    pd.DataFrame(records).to_csv(ws.path('reports/video_audit_sample.csv'), index=False)
    checked = [r for r in records if r['status'] == 'decoded']
    ws.json('reports/video_summary.json', {'sample_episodes': len(records), 'decoded_episodes': len(checked),
            'sample_cameras': [camera], 'download_bytes': consumed, 'video_quota_bytes': budget,
            'frame_count_mismatches': sum(not r['frame_count_match'] for r in checked),
            'decode_errors': sum(r['status']=='decode_error' for r in records),
            'sampling': 'outcome x anomaly flag stratified sample, limited by shard byte budget',
            'scope': 'Converted MP4 and converted timestamps only; not original sensor clocks; no whole-dataset video claim.'})
    print('VIDEO SAMPLE AUDIT COMPLETE', flush=True)
