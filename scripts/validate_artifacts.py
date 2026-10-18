"""Offline release checks. Does not download data or pretend to reproduce metrics."""
import ast
import json
from pathlib import Path
import re
import nbformat
import yaml

root=Path(__file__).resolve().parents[1]
sources=list((root/'src').rglob('*.py'))+[root/'app.py']+list((root/'tests').glob('*.py'))
for path in sources:
    ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
notebooks=list((root/'notebooks').glob('*.ipynb'))
assert notebooks, 'Missing notebooks'
for path in notebooks:
    notebook=nbformat.read(path,as_version=4)
    nbformat.validate(notebook)
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == 'code':
            ast.parse(cell.source,filename=f'{path.name}:cell{index}')
for path in (root/'reports').glob('*.json'):
    json.loads(path.read_text(encoding='utf-8'), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
config=yaml.safe_load((root/'configs/project.yaml').read_text(encoding='utf-8'))
assert all(re.fullmatch('[0-9a-f]{40}',s['revision']) for s in config['sources'].values())
assert config['prefix_frames']==30
for forbidden in ['English Resume.pdf','GEMINI api.docx','.env']:
    assert not (root/forbidden).exists(), f'Private artifact in project: {forbidden}'
print(f'ARTIFACT ACCEPTANCE PASS: {len(sources)} Python files; {len(notebooks)} notebooks; JSON and data pins valid')
