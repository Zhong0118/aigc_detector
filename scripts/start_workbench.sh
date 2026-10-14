#!/usr/bin/env bash
# 启动文本检测工作台。学校机器上 8000 已被占用，所以 API 用 8001。
set -euo pipefail
cd /home/zhongxu/projects/aigc_detector

API_PORT="${API_PORT:-8001}"
DASH_PORT="${DASH_PORT:-8501}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
PY="/home/zhongxu/.conda/envs/zx/bin/python"
export AIGC_API_BASE="http://127.0.0.1:${API_PORT}/api/v1"
export CUDA_VISIBLE_DEVICES="$GPU"

mkdir -p logs
echo "API      -> http://0.0.0.0:${API_PORT}   (health: /health)"
echo "Dashboard-> http://0.0.0.0:${DASH_PORT}"
echo "Using GPU ${GPU}, conda env zx"

"$PY" -m uvicorn api.main:app --host 0.0.0.0 --port "$API_PORT" \
  > logs/api.log 2>&1 &
API_PID=$!
echo "$API_PID" > logs/api.pid

# wait until /health answers
for i in $(seq 1 40); do
  if curl -sf "http://127.0.0.1:${API_PORT}/health" >/dev/null; then
    echo "API ready"
    break
  fi
  sleep 0.5
done

"$PY" -m streamlit run api/dashboard.py \
  --server.address 0.0.0.0 \
  --server.port "$DASH_PORT" \
  --server.headless true \
  > logs/dashboard.log 2>&1 &
DASH_PID=$!
echo "$DASH_PID" > logs/dashboard.pid
echo "Dashboard pid $DASH_PID, API pid $API_PID"
echo "Stop with: kill \$(cat logs/api.pid) \$(cat logs/dashboard.pid)"
