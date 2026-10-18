param([int]$Port = 8000)

. (Join-Path $PSScriptRoot "local_runtime.ps1")
Write-Host "Starting local API: http://127.0.0.1:$Port/health"
& $LocalPython -B -X utf8 -m uvicorn api.main:app --host 127.0.0.1 --port $Port
exit $LASTEXITCODE
