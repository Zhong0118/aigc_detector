$ErrorActionPreference = "Stop"

# Keep model caches inside the project. Do not use --reload because cache writes
# can otherwise restart the API while models are loading.
$env:HF_HOME = "$PWD\models\huggingface"
$env:HF_HUB_CACHE = "$PWD\models\huggingface\hub"
$env:HF_DATASETS_CACHE = "$PWD\models\huggingface\datasets"

$Python = "C:\Users\zx\.conda\envs\dl\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python executable not found: $Python"
}

if ($env:AIGC_SKIP_TEXT_BOOTSTRAP -ne "1") {
    Write-Host "Checking multilingual text resources and calibration artifact v2..."
    & $Python -m scripts.bootstrap_text_resources --profile all --allow-fallback
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

& $Python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
