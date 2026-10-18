"""Allow-listed prefix features: never source, outcome, text, ID, or full duration."""
from __future__ import annotations

import hashlib

import numpy as np

from .data import SIGNALS

EXPECTED_DIMS = dict(zip(SIGNALS, (7, 7, 1, 7, 7, 1)))


def arrays(frame) -> dict[str, np.ndarray]:
    output = {}
    for key in SIGNALS:
        a = np.stack(frame[key].to_numpy()).astype(np.float64)
        if a.ndim == 1 and EXPECTED_DIMS[key] == 1:
            a = a[:, None]
        if a.ndim != 2 or a.shape[1] != EXPECTED_DIMS[key]:
            raise ValueError(f'Invalid shape {key}: {a.shape}')
        output[key] = a
    return output


def content_hash(values: dict[str, np.ndarray], prefix: int | None = None) -> str:
    h = hashlib.sha256()
    for key in SIGNALS:
        a = values[key] if prefix is None else values[key][:prefix]
        a = np.asarray(a, dtype='<f4').copy()
        a[np.isnan(a)] = np.float32(np.nan)
        h.update(key.encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def prefix_features(values: dict[str, np.ndarray], count: int = 30) -> dict[str, float]:
    if min(len(a) for a in values.values()) < count:
        raise ValueError('Episode shorter than fixed prefix')
    features = {}
    for key in SIGNALS:
        a = values[key][:count]
        if key == 'observation.cartesian_position':
            # Position XYZ only: quaternion signs are equivalent, not angular jumps.
            a = a[:, :3]
        safe = np.where(np.isfinite(a), a, np.nan)
        tag = key.replace('.', '_')
        for j in range(a.shape[1]):
            x = safe[:, j]
            finite = x[np.isfinite(x)]
            for statistic, fn in [('mean', np.mean), ('std', np.std), ('min', np.min), ('max', np.max)]:
                features[f'{tag}_{j}_{statistic}'] = float(fn(finite)) if len(finite) else np.nan
            delta = np.diff(x)
            delta = delta[np.isfinite(delta)]
            features[f'{tag}_{j}_change'] = float(x[-1] - x[0])
            features[f'{tag}_{j}_mean_abs_delta'] = float(np.mean(np.abs(delta))) if len(delta) else np.nan
    return features


def trajectory_summary(values: dict[str, np.ndarray]) -> dict[str, float]:
    action = values['action.joint_velocity']
    obs = values['observation.joint_position']
    xyz = values['observation.cartesian_position'][:, :3]
    finite = np.concatenate([a.ravel() for a in values.values()])
    output = {'nonfinite_fraction': float(np.mean(~np.isfinite(finite))),
              'frames': len(action)}
    for tag, a in [('action', action), ('joint', obs), ('position', xyz)]:
        clean = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        speed = np.linalg.norm(np.diff(clean, axis=0), axis=1)
        output[f'{tag}_max_abs'] = float(np.max(np.abs(clean)))
        output[f'{tag}_delta_p95'] = float(np.quantile(speed, .95)) if len(speed) else 0.0
        output[f'{tag}_delta_max'] = float(speed.max()) if len(speed) else 0.0
        output[f'{tag}_path_length'] = float(speed.sum())
    return output
