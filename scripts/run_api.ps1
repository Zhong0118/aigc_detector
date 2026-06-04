$ErrorActionPreference = "Stop"

# 演示/测试后端启动脚本。
# 不使用 --reload，避免 Hugging Face 缓存、训练模型、测试文件变化触发 FastAPI 反复重启。
$env:HF_HOME = "$PWD\models\huggingface"
$env:HF_HUB_CACHE = "$PWD\models\huggingface\hub"
$env:HF_DATASETS_CACHE = "$PWD\models\huggingface\datasets"
$env:TRANSFORMERS_CACHE = "$PWD\models\huggingface\hub"

C:\Users\zx\.conda\envs\dl\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
