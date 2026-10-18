"""Regenerate held-out real-data features and inference in an independent environment."""
from pathlib import Path
import json
import sys

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'src'))
from robot_audit.paths import Workspace
ws=Workspace(root);ws.setup()

import joblib
import numpy as np
import pandas as pd
import yaml
from robot_audit.audit import inspect_episode, feature_columns
from robot_audit.data import SIGNALS

config=yaml.safe_load(ws.path('configs/project.yaml').read_text(encoding='utf-8'))
metrics=json.loads(ws.path('reports/prediction_metrics.json').read_text(encoding='utf-8'))
selection=pd.read_parquet(ws.path('data/processed/selection.parquet'))
features=pd.read_parquet(ws.path('data/processed/features.parquet'))
predictions=pd.read_parquet(ws.path('data/processed/test_predictions.parquet'))
heldout=selection.merge(features[['episode_id','split']],on='episode_id').query("split == 'test'")
checked=[]
for source in ['success','failure']:
    meta=heldout[heldout.source.eq(source)].sort_values('episode_id').iloc[0]
    spec=config['sources'][source]
    shard=f'data/raw/{spec["repo"]}/data/chunk-{int(meta["data/chunk_index"]):03d}/file_{int(meta["data/file_index"]):03d}.parquet'
    frame=pd.read_parquet(ws.path(shard),columns=['episode_index','frame_index','timestamp',*SIGNALS])
    frame=frame[frame.episode_index.eq(meta.episode_index)].sort_values('frame_index',kind='stable')
    audit, regenerated=inspect_episode(meta,frame,config['prefix_frames'])
    assert regenerated is not None and not audit['dimension_error']
    stored=features[features.episode_id.eq(meta.episode_id)].iloc[0]
    assert regenerated['full_hash']==stored.full_hash
    assert regenerated['prefix_hash']==stored.prefix_hash
    columns=feature_columns(features)
    np.testing.assert_allclose([regenerated[c] for c in columns],stored[columns].to_numpy(dtype=float),atol=1e-12,rtol=1e-12,equal_nan=True)
    for name in metrics['models']:
        model=joblib.load(ws.path(f'artifacts/{name}.joblib'))
        values=pd.DataFrame([regenerated])[model['features']]
        probability=float(model['pipeline'].predict_proba(values)[0,1])
        expected=float(predictions[predictions.episode_id.eq(meta.episode_id)&predictions.model.eq(name)].probability_success.iloc[0])
        np.testing.assert_allclose(probability,expected,atol=1e-10,rtol=1e-10)
    checked.append({'source':source,'features_checked':len(columns),'models_checked':len(metrics['models'])})
result={'status':'PASS','environment':'independent Windows virtual environment on D:',
        'python_version':sys.version.split()[0],'real_heldout_samples':len(checked),
        'checks':checked,'feature_tolerance':1e-12,'prediction_tolerance':1e-10,
        'scope':'Regenerates real held-out motion features and all three model predictions; full multi-GB audit is not rerun.'}
ws.json('reports/real_sample_verification.json',result)
print(json.dumps(result,indent=2))
print('CLEAN REAL-SAMPLE ACCEPTANCE PASS')
