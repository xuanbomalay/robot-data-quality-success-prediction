"""Execute the saved-results notebook with an isolated, project-local kernel."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'src'))
from robot_audit.paths import Workspace

ws = Workspace(root)
ws.setup()
os.environ['ROBOT_AUDIT_ROOT'] = str(root)

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--save', action='store_true', help='Save verified outputs in the existing main notebook')
args = parser.parse_args()
path = ws.path('notebooks/00_project_runner.ipynb')
notebook = nbformat.read(path, as_version=4)
assert "MODE='inspect'" in notebook.cells[1].source, 'Verification executes inspect mode only'
source = [{'cell_type': cell.cell_type, 'source': cell.source} for cell in notebook.cells]
source_digest = hashlib.sha256(json.dumps(source, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

# Do not register a kernel in AppData or use a global user's kernel selection.
ws.json('.cache/jupyter/data/kernels/robot-audit/kernel.json', {
    'argv': [sys.executable, '-B', '-m', 'ipykernel_launcher', '-f', '{connection_file}',
             '--HistoryManager.hist_file=:memory:'],
    'display_name': 'Robot audit project environment', 'language': 'python',
})
manager = KernelManager(kernel_name='robot-audit',
                        ip='127.0.0.1',
                        connection_file=str(ws.path('.cache/jupyter/runtime/verification-kernel.json')))
client = NotebookClient(notebook, km=manager, timeout=180, allow_errors=False,
                        resources={'metadata': {'path': str(root)}})
try:
    client.execute()
finally:
    if manager.has_kernel:
        manager.shutdown_kernel(now=True)
    manager.cleanup_resources()

cells = [cell for cell in notebook.cells if cell.cell_type == 'code']
assert all(cell.execution_count is not None for cell in cells), 'Some notebook cells were skipped'
outputs = [output for cell in cells for output in cell.outputs]
assert all(output.output_type != 'error' for output in outputs), 'Notebook contains an error'
images = sum('image/png' in output.get('data', {}) for output in outputs)
assert images >= 2, 'Expected real-data figures did not render into notebook outputs'
if args.save:
    path.write_text(nbformat.writes(notebook), encoding='utf-8', newline='\n')
ws.json('reports/notebook_verification.json', {
    'status': 'PASS', 'mode': 'inspect_saved_results', 'source_sha256': source_digest,
    'code_cells_executed': len(cells), 'embedded_figures': images,
    'scope': 'saved measured reports and figures; no data download, model retraining or simulation rerun',
    'python': sys.version, 'outputs_saved': args.save,
})
print(f'MAIN NOTEBOOK EXECUTION PASS: {len(cells)} cells; {images} embedded figures; no errors')
