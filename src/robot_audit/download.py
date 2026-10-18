"""Pinned public-source downloads, resumable partial files and content hashes."""
from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

import requests

from .paths import Workspace


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def tree(repo: str, revision: str, folder: str) -> list[dict]:
    url = f'https://huggingface.co/api/datasets/{repo}/tree/{revision}/{folder}?recursive=true&limit=1000'
    rows = []
    while url:
        response = requests.get(url, timeout=(15, 90))
        response.raise_for_status()
        rows.extend(x for x in response.json() if x['type'] == 'file')
        url = response.links.get('next', {}).get('url')
    return rows


def fetch_file(ws: Workspace, repo: str, revision: str, entry: dict) -> dict:
    rel = entry['path']
    dest = ws.path(Path('data/raw') / repo / rel)
    expected = int(entry.get('size', 0))
    expected_sha = entry.get('lfs', {}).get('oid')
    sidecar = dest.with_suffix(dest.suffix + '.verified.json')
    if dest.exists() and sidecar.exists() and dest.stat().st_size == expected:
        meta = json.loads(sidecar.read_text(encoding='utf-8'))
        if meta.get('revision') == revision and meta.get('size') == expected:
            return meta
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + '.part')
    url = f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{quote(rel, safe="/")}'
    for attempt in range(5):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > expected:
                raise RuntimeError(f'Partial file exceeds expected size: {rel}')
            if offset != expected:
                headers = {'Range': f'bytes={offset}-'} if offset else {}
                with requests.get(url, headers=headers, stream=True, timeout=(20, 120)) as response:
                    response.raise_for_status()
                    if offset and response.status_code != 206:
                        offset = 0
                    with partial.open('ab' if offset else 'wb') as handle:
                        for chunk in response.iter_content(2 * 1024**2):
                            handle.write(chunk)
                            if handle.tell() > expected:
                                raise RuntimeError(f'Download exceeded expected size: {rel}')
            if partial.stat().st_size != expected:
                raise OSError(f'Incomplete download: {rel}')
            digest = sha256(partial)
            if expected_sha and digest != expected_sha:
                raise RuntimeError(f'SHA256 mismatch: {rel}; preserve partial for inspection.')
            partial.replace(dest)
            meta = {'repo': repo, 'revision': revision, 'path': rel, 'size': expected, 'sha256': digest,
                    'upstream_sha256': expected_sha}
            sidecar.write_text(json.dumps(meta, indent=2), encoding='utf-8')
            return meta
        except (requests.RequestException, OSError):
            if attempt == 4:
                raise
            time.sleep(min(2**attempt, 8))
    raise RuntimeError('Download failed')


def fetch_many(ws: Workspace, repo: str, revision: str, entries: list[dict], workers: int = 4) -> list[dict]:
    additional = sum(max(0, int(x['size']) - (ws.path(Path('data/raw') / repo / x['path']).stat().st_size
                     if ws.path(Path('data/raw') / repo / x['path']).exists() else 0)) for x in entries)
    ws.check_quota(additional)
    result = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_file, ws, repo, revision, entry): entry['path'] for entry in entries}
        for future in as_completed(futures):
            record = future.result()
            result.append(record)
            print(f'DOWNLOAD {len(result)}/{len(entries)} {record["path"]} {record["size"]/1e6:.1f} MB', flush=True)
    return sorted(result, key=lambda x: x['path'])
