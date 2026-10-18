"""Controlled simulation experiment, deliberately separate from DROID claims.

Uses robomimic HDF5 demonstrations with their robosuite 1.5.1 environment config.
No CUDA, PyTorch, or simulator is imported until a cloud training/eval command.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from pathlib import Path

import h5py
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from .download import fetch_file, sha256, tree

OBS_KEYS = ('object', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos')
CONDITIONS = ('clean', 'contaminated', 'random_drop', 'audit_drop', 'oracle_drop')


def policy_quality_features(action: np.ndarray) -> np.ndarray:
    delta = np.diff(action, axis=0)
    jerk = np.diff(delta, axis=0)
    magnitude = np.linalg.norm(delta, axis=1)
    return np.array([np.mean(np.abs(action)), np.std(action), np.max(np.abs(action)),
                     np.mean(magnitude), np.quantile(magnitude, .95), np.max(magnitude),
                     np.mean(np.abs(jerk)), np.mean(np.abs(action) > .98),
                     *np.std(action, axis=0)], dtype=float)


def contaminate(actions: dict[str, np.ndarray], fraction=.1, seed=42):
    rng = np.random.default_rng(seed)
    keys = sorted(actions)
    count = max(1, int(round(len(keys) * fraction)))
    selected = rng.choice(keys, count, replace=False).tolist()
    contaminated = {k: a.copy() for k, a in actions.items()}
    truth = []
    for i, key in enumerate(selected):
        original = actions[key]
        if i % 2 == 0:
            indices = rng.choice(len(original), max(1, len(original)//5), replace=False)
            contaminated[key][indices] = rng.choice([-1., 1.], size=(len(indices), original.shape[1]))
            kind = '20_percent_action_spikes'
        else:
            contaminated[key] = np.roll(original, 5, axis=0)
            kind = 'five_step_circular_action_misalignment'
        truth.append({'episode': key, 'kind': kind,
                      'original_action_sha256': hashlib.sha256(original.tobytes()).hexdigest(),
                      'corrupted_action_sha256': hashlib.sha256(contaminated[key].tobytes()).hexdigest()})
    return contaminated, truth


def select_policy_conditions(clean_actions, fraction=.1, seed=42):
    contaminated, truth = contaminate(clean_actions, fraction, seed)
    keys = sorted(clean_actions)
    count = len(truth)
    # The detector receives actions only. The injection truth is not an argument or a feature.
    x = np.stack([policy_quality_features(contaminated[k]) for k in keys])
    x = RobustScaler().fit_transform(x)
    model = IsolationForest(n_estimators=300, contamination=fraction, random_state=seed, n_jobs=2).fit(x)
    scores = -model.score_samples(x)
    audit_drop = {keys[i] for i in np.argsort(scores, kind='stable')[-count:]}
    random_drop = set(np.random.default_rng(seed + 1).choice(keys, count, replace=False).tolist())
    oracle_drop = {r['episode'] for r in truth}
    conditions = {'clean': dict(clean_actions), 'contaminated': contaminated,
                  'random_drop': {k: v for k, v in contaminated.items() if k not in random_drop},
                  'audit_drop': {k: v for k, v in contaminated.items() if k not in audit_drop},
                  'oracle_drop': {k: v for k, v in contaminated.items() if k not in oracle_drop}}
    detector_report = {'injected_episodes': count, 'train_episodes': len(keys),
                       'retained_per_condition': {k: len(v) for k, v in conditions.items()},
                       'audit_corruption_precision': len(audit_drop & oracle_drop)/count,
                       'audit_corruption_recall': len(audit_drop & oracle_drop)/count,
                       'random_corruption_recall': len(random_drop & oracle_drop)/count,
                       'dropped': {'audit_drop': sorted(audit_drop), 'random_drop': sorted(random_drop), 'oracle_drop': sorted(oracle_drop)}}
    return conditions, truth, detector_report


def prepare(ws, config):
    p = config['policy']
    entry = next(x for x in tree(p['repo'], p['revision'], 'v1.5/lift/ph') if x['path'] == p['file'])
    manifest = fetch_file(ws, p['repo'], p['revision'], entry)
    path = ws.path(Path('data/raw') / p['repo'] / p['file'])
    observations, actions = {}, {}
    with h5py.File(path, 'r') as handle:
        data = handle['data']
        env_meta = json.loads(data.attrs['env_args'])
        train_keys = sorted(x.decode() for x in handle['mask/train'][:])
        valid_keys = sorted(x.decode() for x in handle['mask/valid'][:])
        if set(train_keys) & set(valid_keys):
            raise ValueError('Policy demonstration split leakage')
        for key in train_keys + valid_keys:
            observations[key] = np.concatenate([data[key]['obs'][k][:] for k in OBS_KEYS], axis=1).astype(np.float32)
            actions[key] = data[key]['actions'][:].astype(np.float32)
    conditions, truth, report = select_policy_conditions({k: actions[k] for k in train_keys}, p['corruption_fraction'], config['seed'])
    ws.path('artifacts/policy').mkdir(parents=True, exist_ok=True)
    # Data arrays are not committed. In Colab these remain in the cloud workspace.
    import pickle
    with ws.path('artifacts/policy/prepared.pkl').open('wb') as handle:
        pickle.dump({'observations': observations, 'actions': actions, 'conditions': conditions,
                     'train_keys': train_keys, 'valid_keys': valid_keys, 'env_meta': env_meta}, handle)
    ws.json('artifacts/policy/injection_truth.json', truth)
    ws.json('reports/policy_preparation.json', {**report, 'data': manifest,
              'validation_episodes': len(valid_keys), 'observation_keys': list(OBS_KEYS),
              'environment': env_meta, 'scope': 'simulated Lift; synthetic finite-valued action corruptions'})
    print('POLICY PREPARATION PASS', flush=True)


def _load(ws):
    import pickle
    with ws.path('artifacts/policy/prepared.pkl').open('rb') as handle:
        return pickle.load(handle)


def _network(input_dim, action_dim):
    import torch
    return torch.nn.Sequential(torch.nn.Linear(input_dim, 256), torch.nn.ReLU(),
                               torch.nn.Linear(256, 256), torch.nn.ReLU(),
                               torch.nn.Linear(256, action_dim), torch.nn.Tanh())


def _make_env(prepared, seed, record_video=False):
    import robosuite
    kwargs = copy.deepcopy(prepared['env_meta']['env_kwargs'])
    # Lift 1.5.1 does not expose MujocoEnv's seed keyword in its constructor.
    kwargs.update(has_renderer=False, has_offscreen_renderer=record_video,
                  use_camera_obs=False, hard_reset=False, renderer='mujoco')
    np.random.seed(seed)
    return robosuite.make(prepared['env_meta']['env_name'], **kwargs)


def _reset_seeded(env, seed):
    # The pinned 1.5.1 robot and placement samplers use NumPy's global RNG.
    np.random.seed(seed)
    env.rng.bit_generator.state = np.random.default_rng(seed).bit_generator.state
    return env.reset()


def simulator_smoke(ws, config):
    prepared = _load(ws)
    start = config['policy']['pilot_seeds_start']
    hashes = []
    for repeat in range(2):
        env = _make_env(prepared, start)
        current = []
        try:
            for seed in range(start, start + 3):
                obs = _reset_seeded(env, seed)
                vector = np.concatenate([obs['object-state'] if k == 'object' else obs[k] for k in OBS_KEYS])
                assert vector.shape == (19,) and np.isfinite(vector).all(), 'Simulator observation contract mismatch'
                current.append(hashlib.sha256(np.asarray(env.sim.get_state().flatten(), dtype='<f8').tobytes()).hexdigest())
                env.step(np.full(env.action_dim, .05 * repeat))
        finally:
            env.close()
        hashes.append(current)
    assert hashes[0] == hashes[1], 'Seeded initial states depend on prior rollout history'
    ws.json('reports/policy_environment_smoke.json', {'status': 'PASS', 'observation_dim': 19,
            'eval_seeds': list(range(start, start + 3)), 'initial_state_sha256': hashes[0],
            'repeated_with_different_action_history': True})
    print('SIMULATOR RESET AND ACTION SMOKE PASS', flush=True)


def train_bc(ws, prepared, condition, seed, steps):
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    keys = sorted(prepared['conditions'][condition])
    x = np.concatenate([prepared['observations'][k] for k in keys])
    y = np.concatenate([prepared['conditions'][condition][k] for k in keys])
    # Normalization from original training observations only, fixed across all conditions.
    reference = np.concatenate([prepared['observations'][k] for k in prepared['train_keys']])
    mean, std = reference.mean(axis=0), reference.std(axis=0).clip(.01)
    x = torch.as_tensor((x - mean)/std, dtype=torch.float32, device=device)
    y = torch.as_tensor(y, dtype=torch.float32, device=device)
    model = _network(x.shape[1], y.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    tag = f'{condition}_seed{seed}'
    checkpoint = ws.path(f'artifacts/policy/{tag}.pt')
    start = 0
    if checkpoint.exists():
        previous = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(previous['model'])
        optimizer.load_state_dict(previous['optimizer'])
        start = previous['step']
        if start > steps:
            raise ValueError('Checkpoint is beyond the requested fixed training budget')
        torch.set_rng_state(previous['torch_rng'].cpu())
        if device.type == 'cuda' and previous.get('cuda_rng') is not None:
            torch.cuda.set_rng_state(previous['cuda_rng'].cpu())
    started = time.monotonic()
    loss_value = None
    for step in range(start, steps):
        indices = torch.randint(len(x), (128,), device=device)
        prediction = model(x[indices])
        loss = torch.nn.functional.mse_loss(prediction, y[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (step+1) % 1000 == 0 or step+1 == steps:
            loss_value = float(loss.detach().cpu())
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'step': step+1,
                     'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state() if device.type == 'cuda' else None,
                     'mean': mean, 'std': std, 'input_dim': x.shape[1], 'action_dim': y.shape[1],
                     'condition': condition, 'seed': seed, 'train_loss': loss_value,
                     'architecture': 'MLP 256-256 ReLU, tanh actions; MSE; Adam lr=.001, batch=128'}
            temporary = checkpoint.with_suffix('.tmp')
            torch.save(state, temporary)
            temporary.replace(checkpoint)
            print(f'TRAIN {tag} step={step+1}/{steps} loss={loss_value:.5f} elapsed={time.monotonic()-started:.1f}s device={device}', flush=True)
    return checkpoint


def evaluate(ws, prepared, checkpoint, seed_start, count, horizon=400, record_video=False):
    import torch
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = _network(state['input_dim'], state['action_dim'])
    model.load_state_dict(state['model']); model.eval()
    env = _make_env(prepared, seed_start, record_video)
    output = []
    tag = checkpoint.stem
    dest = ws.path(f'artifacts/policy/{tag}_steps{state["step"]}_eval_{seed_start}.json')
    existing = json.loads(dest.read_text()) if dest.exists() else []
    # Resume evaluation only if the checkpoint step matches; pilot budget extension invalidates results.
    existing = [r for r in existing if r.get('training_steps') == state['step']]
    by_seed = {r['eval_seed']: r for r in existing}
    try:
        for eval_seed in range(seed_start, seed_start + count):
            if eval_seed in by_seed:
                output.append(by_seed[eval_seed]); continue
            obs = _reset_seeded(env, eval_seed)
            initial = np.asarray(env.sim.get_state().flatten(), dtype='<f8')
            initial_hash = hashlib.sha256(initial.tobytes()).hexdigest()
            success = False
            video = []
            for step in range(horizon):
                vector = np.concatenate([obs['object-state'] if k == 'object' else obs[k] for k in OBS_KEYS]).astype(np.float32)
                with torch.inference_mode():
                    action = model(torch.from_numpy((vector - state['mean'])/state['std'])).numpy()
                obs, _, _, _ = env.step(np.clip(action, -1, 1))
                if record_video and len(output) < 2 and step % 2 == 0:
                    video.append(env.sim.render(height=256, width=256, camera_name='agentview')[::-1])
                if env._check_success():
                    success = True
                    break
            if video:
                import imageio.v2 as imageio
                imageio.mimsave(ws.path(f'artifacts/policy/{tag}_{eval_seed}.mp4'), video, fps=10)
            record = {'condition': state['condition'], 'training_seed': state['seed'], 'training_steps': state['step'],
                      'eval_seed': eval_seed, 'success': bool(success), 'steps': step+1, 'initial_state_sha256': initial_hash}
            output.append(record)
            ws.json(dest.relative_to(ws.root), output)
            if len(output) % 10 == 0:
                print(f'EVAL {tag} {len(output)}/{count} successes={sum(r["success"] for r in output)}', flush=True)
    finally:
        env.close()
    return output


def pilot(ws, config):
    p = config['policy']; prepared = _load(ws)
    history = {}
    for path in ws.path('artifacts/policy').glob(f'clean_seed5*_eval_{p["pilot_seeds_start"]}.json'):
        rows = json.loads(path.read_text())
        expected_seeds = set(range(p['pilot_seeds_start'], p['pilot_seeds_start'] + p['pilot_episodes']))
        if len(rows) == p['pilot_episodes'] and {row['eval_seed'] for row in rows} == expected_seeds:
            budgets = {row['training_steps'] for row in rows}
            if len(budgets) != 1:
                raise ValueError('Mixed training budgets in pilot evaluation')
            budget = budgets.pop()
            rate = float(np.mean([row['success'] for row in rows]))
            history[budget] = {'steps': budget, 'success_rate': rate, 'episodes': len(rows), 'passed': rate >= .7}
    resume_step = 0
    previous = ws.path('artifacts/policy/clean_seed5.pt')
    if previous.exists():
        import torch
        resume_step = int(torch.load(previous, map_location='cpu', weights_only=False)['step'])
    for budget in [p['steps'], p['steps']*2, p['steps']*3]:
        if budget < resume_step:
            continue
        checkpoint = train_bc(ws, prepared, 'clean', 5, budget)
        result = evaluate(ws, prepared, checkpoint, p['pilot_seeds_start'], p['pilot_episodes'], p['horizon'])
        rate = float(np.mean([x['success'] for x in result]))
        ws.json('reports/policy_pilot.json', {'steps': budget, 'success_rate': rate,
                'episodes': p['pilot_episodes'], 'required_success_rate': .7, 'passed': rate >= .7,
                'pilot_seed': 5, 'eval_seed_start': p['pilot_seeds_start']})
        history[budget] = {'steps': budget, 'success_rate': rate, 'episodes': len(result), 'passed': rate >= .7}
        ws.json('reports/policy_pilot_history.json', [history[key] for key in sorted(history)])
        print(f'PILOT success={rate:.3f} budget={budget}', flush=True)
        if rate >= .7:
            ws.json('artifacts/policy/frozen_protocol.json', {**p, 'steps': budget, 'pilot_passed': True,
                    'architecture': 'MLP 256-256, Adam .001, batch 128, MSE', 'selection': 'final fixed-step checkpoint',
                    'condition_order': list(CONDITIONS)})
            return
    raise RuntimeError('Clean BC pilot failed; do not run or interpret the cleaning comparison.')


def summarize(ws, config):
    import pandas as pd
    p = config['policy']
    records = []
    for path in ws.path('artifacts/policy').glob('*_eval_10000.json'):
        records.extend(json.loads(path.read_text()))
    if not records:
        raise RuntimeError('No formal policy evaluation results')
    frame = pd.DataFrame(records)
    counts = frame.groupby(['condition', 'training_seed']).size()
    required = {(c, s) for c in CONDITIONS for s in p['train_seeds']}
    if set(counts.index) != required or (counts != p['eval_episodes']).any():
        raise RuntimeError('Formal evaluation incomplete: require five conditions x three seeds x 100 rollouts.')
    protocol = json.loads(ws.path('artifacts/policy/frozen_protocol.json').read_text())
    if set(frame.training_steps) != {protocol['steps']}:
        raise RuntimeError('Formal policies do not share the frozen training budget')
    expected_seeds = set(range(protocol['eval_seeds_start'], protocol['eval_seeds_start'] + protocol['eval_episodes']))
    if set(frame.eval_seed) != expected_seeds:
        raise RuntimeError('Formal evaluation seeds differ from the frozen protocol')
    if frame.groupby('eval_seed').initial_state_sha256.nunique().max() != 1:
        raise RuntimeError('Initial states differ across conditions: paired comparison invalid')
    frame.to_csv(ws.path('reports/policy_rollouts.csv'), index=False)
    pivot = frame.pivot(index=['training_seed', 'eval_seed'], columns='condition', values='success').astype(float)
    rng = np.random.default_rng(config['seed'])
    bootstraps = []
    seeds = p['train_seeds']
    # Crossed clusters: all policies share the same starts, including across training seeds.
    states = np.array(sorted(frame.eval_seed.unique()))
    for _ in range(2000):
        sampled = []
        sampled_states = rng.choice(states, len(states), replace=True)
        for seed in rng.choice(seeds, len(seeds), replace=True):
            sampled.append(pivot.loc[seed].loc[sampled_states])
        bootstraps.append(pd.concat(sampled).mean())
    boot = pd.DataFrame(bootstraps)
    output = {'scope': 'simulation only, injected action corruption only; not DROID policy performance',
              'status': 'complete', 'rollouts': len(frame), 'paired_initial_states': 'PASS',
              'training_steps': protocol['steps'], 'evaluation_horizon': protocol['horizon'],
              'interval_method': 'paired two-way cluster bootstrap: training seeds and shared initial states; 2000 resamples',
              'per_seed': frame.groupby(['condition', 'training_seed']).success.mean().reset_index().to_dict(orient='records'),
              'conditions': {c: {'success_rate': float(pivot[c].mean()),
                             'ci95': [float(boot[c].quantile(.025)), float(boot[c].quantile(.975))]} for c in CONDITIONS},
              'audit_minus_contaminated_percentage_points': float(100*(pivot.audit_drop-pivot.contaminated).mean()),
              'audit_minus_contaminated_ci95_pp': [float(100*(boot.audit_drop-boot.contaminated).quantile(.025)),
                                                  float(100*(boot.audit_drop-boot.contaminated).quantile(.975))],
              'audit_minus_random_percentage_points': float(100*(pivot.audit_drop-pivot.random_drop).mean())}
    ws.json('reports/policy_results.json', output)
    print(json.dumps(output, indent=2), flush=True)
    print('POLICY FIVE-CONDITION ACCEPTANCE PASS', flush=True)


def policy_main(ws, config, stage, seed=None, condition=None):
    if stage == 'prepare':
        prepare(ws, config)
    elif stage == 'smoke':
        simulator_smoke(ws, config)
    elif stage == 'pilot':
        pilot(ws, config)
    elif stage == 'summarize':
        summarize(ws, config)
    elif stage == 'run':
        frozen = json.loads(ws.path('artifacts/policy/frozen_protocol.json').read_text())
        if not frozen['pilot_passed']:
            raise RuntimeError('Pass pilot first')
        prepared = _load(ws)
        for c in ([condition] if condition else CONDITIONS):
            for s in ([seed] if seed is not None else frozen['train_seeds']):
                checkpoint = train_bc(ws, prepared, c, s, frozen['steps'])
                evaluate(ws, prepared, checkpoint, frozen['eval_seeds_start'], frozen['eval_episodes'], frozen['horizon'])
        if condition is None and seed is None:
            summarize(ws, config)
