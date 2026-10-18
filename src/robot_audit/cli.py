from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from .paths import Workspace


def main():
    parser = argparse.ArgumentParser(description='Robot quality audit: explicit workspace, pinned data, held-out evaluation')
    parser.add_argument('--root', default=os.environ.get('ROBOT_AUDIT_ROOT', str(Path.cwd())))
    parser.add_argument('--config', default='configs/project.yaml')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('preflight')
    fetch = sub.add_parser('fetch')
    fetch.add_argument('--small', action='store_true')
    fetch.add_argument('--metadata-only', action='store_true')
    sub.add_parser('audit')
    sub.add_parser('detect')
    model = sub.add_parser('train')
    model.add_argument('--bootstrap', type=int, default=1000)
    video = sub.add_parser('video')
    video.add_argument('--episodes', type=int, default=30)
    sub.add_parser('report')
    sub.add_parser('demo')
    sub.add_parser('test')
    policy = sub.add_parser('policy')
    policy.add_argument('stage', choices=['prepare', 'smoke', 'pilot', 'run', 'summarize'])
    policy.add_argument('--seed', type=int)
    policy.add_argument('--condition', choices=['clean', 'contaminated', 'random_drop', 'audit_drop', 'oracle_drop'])
    args = parser.parse_args()
    ws = Workspace(args.root)
    ws.setup()
    import yaml
    config = yaml.safe_load(ws.path(args.config).read_text(encoding='utf-8'))
    ws.quota_bytes = int(config['quota_gb'] * 1024**3)
    if args.command == 'preflight':
        result = ws.check_quota()
        result.update({'python': sys.version, 'executable': sys.executable, 'platform': platform.platform()})
        ws.json('reports/internal/environment.json', result)
        freeze = subprocess.run([sys.executable, '-B', '-m', 'pip', 'freeze'], capture_output=True, text=True, check=True)
        ws.path('requirements-lock-windows.txt').write_text(freeze.stdout, encoding='utf-8')
        print(json.dumps(result, indent=2))
    elif args.command == 'fetch':
        from .data import fetch_metadata, select_episodes, fetch_controls
        fetch_metadata(ws, config)
        select_episodes(ws, config, small=args.small)
        if not args.metadata_only:
            fetch_controls(ws, config)
    elif args.command == 'audit':
        from .audit import audit
        audit(ws, config)
    elif args.command == 'detect':
        from .audit import detect_anomalies
        detect_anomalies(ws, config)
    elif args.command == 'train':
        from .modeling import train_predictors
        train_predictors(ws, config, args.bootstrap)
    elif args.command == 'video':
        from .video import audit_videos
        audit_videos(ws, config, args.episodes)
    elif args.command == 'report':
        from .reporting import create_reports
        create_reports(ws, config)
    elif args.command == 'demo':
        subprocess.run([sys.executable, '-B', '-m', 'streamlit', 'run', str(ws.path('app.py')),
                        '--server.address=127.0.0.1', '--server.port=8501', '--server.headless=true',
                        '--browser.gatherUsageStats=false'], check=True)
    elif args.command == 'test':
        subprocess.run([sys.executable, '-B', '-m', 'pytest', str(ws.path('tests')),
                        '--basetemp', str(ws.path('.cache/pytest-temp'))], check=True)
    elif args.command == 'policy':
        from .policy import policy_main
        policy_main(ws, config, args.stage, seed=args.seed, condition=args.condition)


if __name__ == '__main__':
    main()
