"""Build readable runner notebooks and a self-contained, hash-labelled Colab bundle."""
from pathlib import Path
import base64
import hashlib
import io
import zipfile
import nbformat as nbf

root=Path(__file__).resolve().parents[1]
out=root/'notebooks';out.mkdir(exist_ok=True)

def write_notebook(notebook, destination):
    # Stable cell IDs keep rebuilds reviewable and make the exported artifact reproducible.
    for index, cell in enumerate(notebook.cells):
        cell.id=hashlib.sha256(f'{index}:{cell.cell_type}:{cell.source}'.encode()).hexdigest()[:12]
    destination.write_text(nbf.writes(notebook),encoding='utf-8',newline='\n')


main=nbf.v4.new_notebook()
main.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}}
main.cells=[nbf.v4.new_markdown_cell('''# Robot Data Quality & Success Prediction

真实机器人数据质量审计、固定前缀成功预测，以及独立的仿真清洗实验。

先按 README 在 D 盘完成依赖安装。默认 **inspect** 模式读取已生成的实测结果；改为 **reproduce** 会下载并重跑真实数据流程。没有输出时会明确报错，不生成虚构结果。完整实验、抽样检查与 CI 合成测试的证据层级分别记录。'''),nbf.v4.new_code_cell('''import os, sys, subprocess, json
from pathlib import Path
ROOT=Path(os.environ.get('ROBOT_AUDIT_ROOT',Path.cwd())).resolve()
if not (ROOT/'configs/project.yaml').exists() and (ROOT.parent/'configs/project.yaml').exists(): ROOT=ROOT.parent
assert (ROOT/'configs/project.yaml').exists(), 'Run from the project folder'
sys.path.insert(0,str(ROOT/'src'))
from robot_audit.paths import Workspace
ws=Workspace(ROOT);ws.setup()
MODE='inspect'  # change to 'reproduce' for a full real-data rerun
assert MODE in ('inspect','reproduce')
def run(*arguments):
    subprocess.run([sys.executable,'-B','-m','robot_audit','--root',str(ROOT),*arguments],check=True,cwd=ROOT)
print('Workspace:',ROOT)
'''),nbf.v4.new_markdown_cell('## 1. 数据与审计\n固定数据提交、标签冲突隔离、轨迹去重、结构与转换后时间戳检查。'),nbf.v4.new_code_cell('''if MODE=='reproduce':
    for command in ['fetch','audit','detect']: run(command)
audit=json.loads((ROOT/'reports/audit_summary.json').read_text(encoding='utf-8'))
assert audit['leakage_checks']=='PASS'
print(json.dumps(audit,indent=2,ensure_ascii=False))
'''),nbf.v4.new_markdown_cell('## 2. 成功预测\n只用前 30 帧运动特征；模型与阈值依验证集选择，测试集按采集者保留。'),nbf.v4.new_code_cell('''if MODE=='reproduce': run('train')
metrics=json.loads((ROOT/'reports/prediction_metrics.json').read_text(encoding='utf-8'))
import pandas as pd
display(pd.DataFrame({name:record['test'] for name,record in metrics['models'].items()}).T)
print('Validation-selected model:',metrics['selected_model'])
'''),nbf.v4.new_markdown_cell('## 3. 视频抽样与报告\n视频只检查一个明确记录的样本。没有原始传感器时钟，不能据此判断原始时钟漂移。'),nbf.v4.new_code_cell('''if MODE=='reproduce':
    run('video','--episodes','30')
    run('report')
from IPython.display import Image, Markdown, display
import re
report=(ROOT/'reports/research_report.md').read_text(encoding='utf-8')
display(Markdown(re.sub(r'!\\[[^\\]]*\\]\\(figures/[^)]+\\)','',report)))
for name in ['audit','prediction']:
    display(Image(filename=str(ROOT/f'reports/figures/{name}.png')))
print('REAL-DATA REPORT INSPECTION PASS' if MODE=='inspect' else 'REAL-DATA SEQUENTIAL RUN PASS')
'''),nbf.v4.new_markdown_cell('## 4. 策略实验\n用 `01_colab_policy.ipynb` 在 Colab 独立运行；达到五组 × 三种子 × 100 次评测并验证初始状态后，才生成 `policy_results.json`。免费配额不足时保留断点，不能将准备完成写成实验完成。')]
main.cells.append(nbf.v4.new_code_cell('''policy_path=ROOT/'reports/policy_results.json'
if policy_path.exists():
    policy=json.loads(policy_path.read_text(encoding='utf-8'))
    assert policy['status']=='complete' and policy['rollouts']==1500
    display(pd.DataFrame(policy['conditions']).T)
    display(Image(filename=str(ROOT/'reports/figures/policy.png')))
    print('POLICY RESULT INSPECTION PASS')
else:
    print('POLICY EVALUATION PENDING: no improvement claim')
'''))
write_notebook(main,out/'00_project_runner.ipynb')

buffer=io.BytesIO()
with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
    for rel in ['src/robot_audit/__init__.py','src/robot_audit/paths.py','src/robot_audit/download.py','src/robot_audit/policy.py','configs/project.yaml']:
        entry=zipfile.ZipInfo(rel,date_time=(2026,1,1,0,0,0))
        entry.compress_type=zipfile.ZIP_DEFLATED
        entry.create_system=3
        entry.external_attr=0o100644 << 16
        archive.writestr(entry,(root/rel).read_text(encoding='utf-8').replace('\r\n','\n').encode())
payload=base64.b64encode(buffer.getvalue()).decode()
digest=hashlib.sha256(buffer.getvalue()).hexdigest()
bootstrap=f'''import os, sys, subprocess, base64, io, zipfile, hashlib
from pathlib import Path
print('Cloud Python:',sys.version,flush=True)
assert (3,10) <= sys.version_info[:2] <= (3,12), 'Select Colab runtime version 2026.07 (Python 3.12) before installing these pinned dependencies.'
loaded_numpy_version=getattr(sys.modules.get('numpy'),'__version__',None)
os.environ['MUJOCO_GL']='egl'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['PIP_CACHE_DIR']='/content/robot_audit_cache/pip'
os.environ['NUMBA_CACHE_DIR']='/content/robot_audit_cache/numba'
for key in ['PIP_CACHE_DIR','NUMBA_CACHE_DIR']: Path(os.environ[key]).mkdir(parents=True,exist_ok=True)
print('Installing pinned simulation dependencies...',flush=True)
subprocess.run([sys.executable,'-m','pip','install','--only-binary=numpy,scipy,h5py,scikit-learn,mujoco','numpy==1.26.4','robosuite==1.5.1','mujoco==3.3.7','h5py==3.14.0','scikit-learn==1.6.1','requests==2.32.5','PyYAML==6.0.2','imageio==2.37.0','imageio-ffmpeg==0.6.0'],check=True)
from importlib.metadata import version
if loaded_numpy_version and loaded_numpy_version != version('numpy'):
    raise RuntimeError('Dependencies installed. Restart the Colab session, then run all cells to load the pinned NumPy ABI consistently.')
source_root=Path('/content/robot_audit_source');source_root.mkdir(exist_ok=True)
bundle=base64.b64decode({payload!r})
assert hashlib.sha256(bundle).hexdigest()=={digest!r}
zipfile.ZipFile(io.BytesIO(bundle)).extractall(source_root)
sys.path.insert(0,str(source_root/'src'))
print('Source bundle verified; importing simulator...',flush=True)
import torch, robosuite, mujoco, yaml, importlib
import robot_audit.policy as policy_module
importlib.reload(policy_module)
print({{'python':sys.version,'torch':torch.__version__,'robosuite':robosuite.__version__,'mujoco':mujoco.__version__,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}})
'''
mount='''from google.colab import drive
drive.mount('/content/drive',timeout_ms=300000)
from robot_audit.paths import Workspace
cloud_root=Path('/content/drive/MyDrive/robot-data-quality-success-prediction')
cloud_root.mkdir(exist_ok=True)
ws=Workspace(cloud_root);ws.setup()
config=yaml.safe_load((source_root/'configs/project.yaml').read_text())
from robot_audit.policy import policy_main
ws.json('reports/cloud_source_manifest.json',{'bundle_sha256':hashlib.sha256(bundle).hexdigest(),
    'source_sha256':{str(p.relative_to(source_root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_root.rglob('*.py')},
    'python':sys.version,'torch':torch.__version__,'device':'cuda' if torch.cuda.is_available() else 'cpu'})
ws.path('reports/cloud_requirements.txt').write_text(subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True,check=True).stdout)
print('CLOUD STORAGE PREFLIGHT PASS',cloud_root)
'''
nb=nbf.v4.new_notebook()
nb.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'colab':{'name':'Robot_Data_Quality_Policy_Experiment.ipynb'}}
nb.cells=[nbf.v4.new_markdown_cell(f'''# Robot data quality: controlled Lift policy experiment

五组 × 三个训练随机种子 × 100 次相同初始状态评测。先通过独立 pilot 再冻结步数。BC 使用本项目 PyTorch 实现，数据来自 robomimic，环境为 robosuite 1.5.1。此实验不代表真实 DROID 机器人性能。

运行前在“代码执行程序 → 更改运行时类型”中选择 CPU 和运行时版本 **2026.07**，以固定 Python 3.12。本轮实测使用免费 CPU；复现同一实验请保持同一后端。安装器拒绝在 Python 3.13 上编译旧版 NumPy。

首次安装如果更换了已载入的 NumPy，安装单元会明确要求“重启会话，然后全部运行”。依赖保留在当前云端虚拟机中，重启后顺序运行即可；不要在混合 ABI 的旧内核里继续执行。

训练与安装仅在 Colab；断点保存到本账号云端硬盘项目目录。免费 CPU 不可用时停止并保留状态，不购买算力。改用 GPU 时应另建实验目录并重新运行，避免在同一次对照中混合计算后端。

源码包由 `scripts/build_notebooks.py` 从仓库中的可读模块生成，SHA256 为 `{digest}`。Base64 仅用于在 GitHub 发布前让笔记本自包含，不包含数据、密钥或个人文件。'''),nbf.v4.new_code_cell(bootstrap),nbf.v4.new_code_cell(mount),nbf.v4.new_code_cell("policy_main(ws, config, 'prepare')\npolicy_main(ws, config, 'pilot')"),nbf.v4.new_code_cell("policy_main(ws, config, 'run')"),nbf.v4.new_code_cell("policy_main(ws, config, 'summarize')\nprint('COLAB SEQUENTIAL RUN ACCEPTANCE PASS')\nprint(ws.path('reports/policy_results.json').read_text())")]
nb.cells[3].source="policy_main(ws, config, 'prepare')\npolicy_main(ws, config, 'smoke')\npolicy_main(ws, config, 'pilot')"
write_notebook(nb,out/'01_colab_policy.ipynb')
print('NOTEBOOK BUILD PASS',digest)
