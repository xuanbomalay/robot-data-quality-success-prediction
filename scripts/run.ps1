param([Parameter(ValueFromRemainingArguments=$true)][string[]]$TaskArgs)
$ErrorActionPreference = 'Stop'
$projectRoot = [System.IO.Path]::GetFullPath((Split-Path $PSScriptRoot -Parent))
if (-not $projectRoot.StartsWith('D:\', [StringComparison]::OrdinalIgnoreCase)) { throw 'This project must run from D: on Windows.' }
$env:ROBOT_AUDIT_ROOT = $projectRoot
$cacheRoot = Join-Path $projectRoot '.cache'
$mapping = @{
 TEMP='temp'; TMP='temp'; PIP_CACHE_DIR='pip'; UV_CACHE_DIR='uv';
 HF_HOME='huggingface'; HF_HUB_CACHE='huggingface/hub'; HF_XET_CACHE='huggingface/xet';
 HF_DATASETS_CACHE='huggingface/datasets'; HF_LEROBOT_HOME='lerobot'; TORCH_HOME='torch';
 XDG_CACHE_HOME='xdg'; MPLCONFIGDIR='matplotlib'; NUMBA_CACHE_DIR='numba';
 IPYTHONDIR='ipython'; JUPYTER_CONFIG_DIR='jupyter/config'; JUPYTER_RUNTIME_DIR='jupyter/runtime';
 PYTHONUSERBASE='python-user'; PYTHONPYCACHEPREFIX='pycache'
}
foreach ($entry in $mapping.GetEnumerator()) {
 $target = [IO.Path]::GetFullPath((Join-Path $cacheRoot $entry.Value))
 if (-not $target.StartsWith($projectRoot + '\', [StringComparison]::OrdinalIgnoreCase)) { throw "Unsafe path: $target" }
 New-Item -ItemType Directory -Path $target -Force | Out-Null
 [Environment]::SetEnvironmentVariable($entry.Key, $target, 'Process')
}
$env:PYTHONUTF8='1'; $env:PYTHONIOENCODING='utf-8'; $env:PYTHONDONTWRITEBYTECODE='1'; $env:PIP_DISABLE_PIP_VERSION_CHECK='1'
$env:PYTHONPATH = Join-Path $projectRoot 'src'
$python = Join-Path $projectRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Run scripts/bootstrap.ps1 first.' }
Push-Location $projectRoot
try { & $python -B -m robot_audit @TaskArgs; if ($LASTEXITCODE -ne 0) { throw "Command failed: $LASTEXITCODE" } }
finally { Pop-Location }
