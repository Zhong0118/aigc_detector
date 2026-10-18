param(
    [int]$Port = 8501,
    [int]$ApiPort = 8000
)

. (Join-Path $PSScriptRoot "local_runtime.ps1")
$env:AIGC_API_BASE = "http://127.0.0.1:$ApiPort/api/v1"
Write-Host "Starting local dashboard: http://127.0.0.1:$Port"
& $LocalPython -B -X utf8 -m streamlit run api/dashboard.py --server.address 127.0.0.1 --server.port $Port --server.headless true --browser.gatherUsageStats false
exit $LASTEXITCODE
