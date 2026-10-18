from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .audit import feature_columns


def choose_failure_threshold(y_success, p_failure) -> float:
    y = 1 - np.asarray(y_success)
    candidates = np.linspace(.05, .95, 91)
    # Maximise failure F1 on validation, resolving ties toward the lower threshold.
    scores = []
    for threshold in candidates:
        pred = p_failure >= threshold
        precision = precision_score(y, pred, zero_division=0)
        recall = recall_score(y, pred, zero_division=0)
        scores.append(2 * precision * recall / max(precision + recall, 1e-12))
    return float(candidates[int(np.argmax(scores))])


def metrics(y, p, failure_threshold: float) -> dict:
    y = np.asarray(y)
    failure_prediction = (1 - p) >= failure_threshold
    return {'roc_auc_success': float(roc_auc_score(y, p)),
            'pr_auc_success': float(average_precision_score(y, p)),
            'pr_auc_failure': float(average_precision_score(1-y, 1-p)),
            'brier_score': float(brier_score_loss(y, p)),
            'failure_recall': float(recall_score(1-y, failure_prediction, zero_division=0)),
            'failure_precision': float(precision_score(1-y, failure_prediction, zero_division=0)),
            'failure_threshold': float(failure_threshold),
            'confusion_matrix_failure_positive': confusion_matrix(1-y, failure_prediction, labels=[0, 1]).tolist()}


def grouped_bootstrap(predictions: pd.DataFrame, repetitions: int = 1000, seed: int = 42) -> dict:
    groups = predictions.collector.unique()
    index = {g: np.where(predictions.collector.to_numpy() == g)[0] for g in groups}
    rng = np.random.default_rng(seed)
    records = []
    for _ in range(repetitions):
        chosen = rng.choice(groups, len(groups), replace=True)
        subset = predictions.iloc[np.concatenate([index[g] for g in chosen])]
        if subset.success.nunique() < 2:
            continue
        m = metrics(subset.success.to_numpy(), subset.probability_success.to_numpy(), float(subset.failure_threshold.iloc[0]))
        records.append({k: m[k] for k in ['roc_auc_success', 'pr_auc_failure', 'failure_recall']})
    frame = pd.DataFrame(records)
    return {k: [float(frame[k].quantile(.025)), float(frame[k].quantile(.975))] for k in frame}


def train_predictors(ws, config: dict, bootstrap: int = 1000) -> dict:
    data = pd.read_parquet(ws.path('data/processed/features.parquet'))
    columns = feature_columns(data)
    subsets = {name: data[data.split == name] for name in ['train', 'validation', 'test']}
    if min(len(g) for g in subsets.values()) < 10:
        raise ValueError('Too few episodes for meaningful model comparison')
    models = {
        'constant_prior': Pipeline([('imputer', SimpleImputer(keep_empty_features=True)), ('model', DummyClassifier(strategy='prior'))]),
        'logistic_regression': Pipeline([('imputer', SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)),
                                       ('scale', StandardScaler()),
                                       ('model', LogisticRegression(C=.1, max_iter=2000, random_state=config['seed']))]),
        'lightgbm': Pipeline([('imputer', SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)),
                             ('model', LGBMClassifier(n_estimators=300, learning_rate=.03, num_leaves=15,
                              min_child_samples=80, reg_lambda=10, colsample_bytree=.8, subsample=1.0,
                              random_state=config['seed'], n_jobs=4, verbosity=-1, deterministic=True, force_col_wise=True))]),
    }
    output = {'protocol': {'feature_count': len(columns), 'prefix_frames': config['prefix_frames'],
              'split': 'collector-grouped 5/7 train, 1/7 validation, 1/7 test; actual fractions reported',
              'threshold': 'failure F1 selected using validation only', 'bootstrap': f'{bootstrap} resamples of test collectors',
              'label': 'eventual teleoperated demonstration success; not autonomous policy reliability',
              'preprocessing_fit': 'train only', 'class_prevalence_success': {s: float(g.success.mean()) for s, g in subsets.items()}}, 'models': {}}
    all_predictions = []
    for name, model in models.items():
        train, validation, test = subsets['train'], subsets['validation'], subsets['test']
        model.fit(train[columns], train.success)
        pv = model.predict_proba(validation[columns])[:, 1]
        threshold = choose_failure_threshold(validation.success, 1-pv)
        pt = model.predict_proba(test[columns])[:, 1]
        record = {'validation': metrics(validation.success.to_numpy(), pv, threshold),
                  'test': metrics(test.success.to_numpy(), pt, threshold)}
        predictions = test[['episode_id', 'collector', 'success']].copy()
        predictions['probability_success'] = pt
        predictions['failure_threshold'] = threshold
        predictions['model'] = name
        record['test_ci95'] = grouped_bootstrap(predictions, bootstrap, config['seed'])
        output['models'][name] = record
        all_predictions.append(predictions)
        joblib.dump({'pipeline': model, 'features': columns, 'failure_threshold': threshold,
                     'prefix_frames': config['prefix_frames']}, ws.path(f'artifacts/{name}.joblib'))
        print(f'MODEL {name}: test AUC={record["test"]["roc_auc_success"]:.4f}', flush=True)
    # Select the presentation model without inspecting the held-out test ranking.
    chosen = max(['logistic_regression', 'lightgbm'], key=lambda n: output['models'][n]['validation']['pr_auc_failure'])
    output['selected_model'] = chosen
    output['selection_metric'] = 'validation PR-AUC for failure'
    selected = models[chosen]
    importance = permutation_importance(selected, subsets['validation'][columns], subsets['validation'].success,
                                       scoring='roc_auc', n_repeats=3, random_state=config['seed'], n_jobs=1)
    pd.DataFrame({'feature': columns, 'validation_permutation_auc_drop': importance.importances_mean,
                  'std': importance.importances_std}).sort_values('validation_permutation_auc_drop', ascending=False).to_csv(
                      ws.path('reports/feature_importance.csv'), index=False)
    pd.concat(all_predictions).to_parquet(ws.path('data/processed/test_predictions.parquet'), index=False)
    ws.json('reports/prediction_metrics.json', output)
    return output
