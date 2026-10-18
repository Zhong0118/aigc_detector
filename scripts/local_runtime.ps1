$ErrorActionPreference = "Stop"

$LocalProjectRoot = Split-Path -Parent $PSScriptRoot
$LocalPython = Join-Path $LocalProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $LocalPython)) {
    throw "Project Python not found: $LocalPython"
}

# The project environment shares the existing Miniconda installation's GPU
# packages. Include its DLL directory in this process's search path.
$VenvConfig = Join-Path $LocalProjectRoot ".venv\pyvenv.cfg"
$BaseSetting = Select-String -LiteralPath $VenvConfig -Pattern '^home\s*=\s*(.+)$'
if ($BaseSetting) {
    $BasePythonDirectory = $BaseSetting.Matches[0].Groups[1].Value.Trim()
    $env:PATH = "$BasePythonDirectory;$BasePythonDirectory\Library\bin;$env:PATH"
}

$env:PYTHONUTF8 = "1"
$env:PYTHONDONTWRITEBYTECODE = "1"
$LocalConfigPath = Join-Path $LocalProjectRoot "config.local.yaml"
if (Test-Path -LiteralPath $LocalConfigPath) {
    $env:AIGC_CONFIG_PATH = $LocalConfigPath
}
$env:HF_HOME = Join-Path $LocalProjectRoot "models\huggingface"
$env:HF_HUB_CACHE = Join-Path $env:HF_HOME "hub"
$env:HF_DATASETS_CACHE = Join-Path $env:HF_HOME "datasets"

# First verify the local application. Model downloads are a separate step.
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:TEMP = Join-Path $LocalProjectRoot "work\tmp"
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Path $env:TEMP -Force | Out-Null
Set-Location -LiteralPath $LocalProjectRoot
