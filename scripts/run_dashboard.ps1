$ErrorActionPreference = "Stop"

# Streamlit 前端启动脚本。
$env:AIGC_API_BASE = "http://127.0.0.1:8000/api/v1"

C:\Users\zx\.conda\envs\dl\python.exe -m streamlit run api/dashboard.py --server.address 127.0.0.1 --server.port 8501
