param(
    [string]$RunConfig = "configs/text_attribution/runs/local_pilot.json",
    [string]$EnvFile = ".env.local",
    [string]$PythonExe = "python",
    [switch]$Confirm
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $RunConfig)) {
    throw "Run config not found: $RunConfig"
}
if (-not (Test-Path -LiteralPath $EnvFile)) {
    throw "Environment file not found: $EnvFile"
}

New-Item -ItemType Directory -Path "logs" -Force | Out-Null
$runName = [System.IO.Path]::GetFileNameWithoutExtension($RunConfig)
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$logPath = "logs/zhizengzeng-$runName-$timestamp.log"
$arguments = @(
    "-m", "scripts.collect_attribution_run",
    "--config", $RunConfig,
    "--env-file", $EnvFile
)
if ($Confirm) {
    $arguments += "--confirm"
} else {
    $arguments += "--dry-run"
}

& $PythonExe @arguments 2>&1 | Tee-Object -FilePath $logPath
exit $LASTEXITCODE
