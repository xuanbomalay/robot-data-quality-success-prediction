param([string]$BasePython = $env:ROBOT_AUDIT_BASE_PYTHON)
$ErrorActionPreference='Stop'
$projectRoot = [IO.Path]::GetFullPath((Split-Path $PSScriptRoot -Parent))
if (-not $projectRoot.StartsWith('D:\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Project environment must be on D:.' }
if (-not $BasePython) {
 $existingPython = Join-Path (Split-Path $projectRoot -Parent) '.venv/Scripts/python.exe'
 if (Test-Path -LiteralPath $existingPython) { $BasePython = $existingPython }
 else { throw 'Specify -BasePython with an existing Python interpreter; no system installation is performed.' }
}
& $BasePython -B -c "import sys; assert sys.version_info[:2] == (3, 12), 'Use an existing Python 3.12 interpreter for the measured Windows environment.'"
if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 is required; no system installation was attempted.' }
foreach ($folder in @('.cache/temp','.cache/pip','.venv','data/raw','data/processed','artifacts','reports','logs')) {
 New-Item -ItemType Directory -Path (Join-Path $projectRoot $folder) -Force | Out-Null
}
$env:TEMP=Join-Path $projectRoot '.cache/temp'; $env:TMP=$env:TEMP
$env:PIP_CACHE_DIR=Join-Path $projectRoot '.cache/pip'
$env:PYTHONUTF8='1'; $env:PYTHONDONTWRITEBYTECODE='1'; $env:PIP_DISABLE_PIP_VERSION_CHECK='1'
if (-not (Test-Path -LiteralPath (Join-Path $projectRoot '.venv/Scripts/python.exe'))) {
 & $BasePython -B -m venv (Join-Path $projectRoot '.venv')
 if ($LASTEXITCODE -ne 0) { throw 'Could not create the project environment on D:.' }
}
$requirements = Join-Path $projectRoot 'requirements-lock-windows.txt'
if (-not (Test-Path -LiteralPath $requirements)) { $requirements = Join-Path $projectRoot 'requirements.txt' }
Write-Output ('Installing project dependencies from ' + (Split-Path $requirements -Leaf))
& (Join-Path $projectRoot '.venv/Scripts/python.exe') -B -m pip install -r $requirements
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
