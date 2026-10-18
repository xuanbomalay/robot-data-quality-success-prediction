"""All project writes are rooted explicitly; Windows writes to C: are rejected."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

CACHE_ENV = {
    'TEMP': 'temp', 'TMP': 'temp', 'PIP_CACHE_DIR': 'pip', 'UV_CACHE_DIR': 'uv',
    'HF_HOME': 'huggingface', 'HF_HUB_CACHE': 'huggingface/hub',
    'HF_XET_CACHE': 'huggingface/xet', 'HF_DATASETS_CACHE': 'huggingface/datasets',
    'HF_LEROBOT_HOME': 'lerobot', 'TORCH_HOME': 'torch', 'XDG_CACHE_HOME': 'xdg',
    'MPLCONFIGDIR': 'matplotlib', 'NUMBA_CACHE_DIR': 'numba', 'IPYTHONDIR': 'ipython',
    'JUPYTER_CONFIG_DIR': 'jupyter/config', 'JUPYTER_RUNTIME_DIR': 'jupyter/runtime',
    'JUPYTER_DATA_DIR': 'jupyter/data',
    'PYTHONUSERBASE': 'python-user', 'PYTHONPYCACHEPREFIX': 'pycache',
}


def canonical(path: Path) -> Path:
    resolved = path.resolve()
    text = str(resolved)
    # Windows may return the extended-length spelling for a missing child while
    # another download creates its parent. Normalize spelling after resolving
    # junctions; this does not relax the root/drive check.
    if os.name == 'nt' and text.startswith('\\\\?\\') and not text.startswith('\\\\?\\UNC\\'):
        return Path(text[4:])
    return resolved


def require_cloud_mount(root: Path) -> None:
    # A stale /content/drive directory can survive a failed DriveFS connection.
    # Directory existence alone must never be accepted as durable checkpoint storage.
    if root.as_posix().startswith('/content/drive/') and not Path('/content/drive').is_mount():
        raise RuntimeError('Google Drive is not mounted; refusing to write checkpoints to temporary VM storage.')


class Workspace:
    def __init__(self, root: str | Path | None = None, quota_gb: float = 30):
        self.root = canonical(Path(root or os.environ.get('ROBOT_AUDIT_ROOT', Path.cwd())))
        if os.name == 'nt' and self.root.drive.upper() != 'D:':
            raise ValueError('Windows project writes are restricted to D:.')
        if os.name != 'nt' and not str(self.root).startswith(('/content/', '/tmp/', '/home/', '/workspace/', '/mnt/')):
            raise ValueError('Use an explicit cloud/CI workspace, not a system directory.')
        require_cloud_mount(self.root)
        self.quota_bytes = int(quota_gb * 1024**3)

    def path(self, relative: str | Path) -> Path:
        require_cloud_mount(self.root)
        path = canonical(self.root / relative)
        if not path.is_relative_to(self.root):
            raise ValueError(f'Path escapes project root: {relative}; resolved={path!s}; root={self.root!s}')
        if os.name == 'nt' and path.drive.upper() != 'D:':
            raise ValueError('Resolved write path is outside D:.')
        return path

    def setup(self) -> None:
        for name, rel in CACHE_ENV.items():
            target = self.path(Path('.cache') / rel)
            target.mkdir(parents=True, exist_ok=True)
            os.environ[name] = str(target)
        os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
        os.environ['PYTHONUTF8'] = '1'
        tempfile.tempdir = str(self.path('.cache/temp'))
        for rel in ('data/raw', 'data/processed', 'artifacts', 'reports/internal', 'logs'):
            self.path(rel).mkdir(parents=True, exist_ok=True)

    def usage(self) -> int:
        return sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file())

    def check_quota(self, additional: int = 0) -> dict:
        used = self.usage()
        free = shutil.disk_usage(self.root).free
        if used + additional > self.quota_bytes:
            raise RuntimeError(f'30 GiB project quota: used={used}, requested={additional}')
        if additional + 2 * 1024**3 > free:
            raise RuntimeError('Insufficient free space; keep 2 GiB headroom.')
        return {'root': str(self.root), 'used_bytes': used, 'limit_bytes': self.quota_bytes,
                'free_bytes': free, 'cache_paths': {k: os.environ.get(k) for k in CACHE_ENV}}

    def json(self, relative: str | Path, value) -> None:
        path = self.path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + '.tmp')
        tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str, allow_nan=False), encoding='utf-8')
        tmp.replace(path)
