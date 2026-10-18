# 本地接手：先打开页面，再接入图片检测

项目目录：`E:\aigc_detector-server-20261015`。

## 1. 当前使用的环境

- 已有 Python：`D:\miniconda3\python.exe`。
- 项目 Python：`.venv\Scripts\python.exe`。
- `.venv` 可以读取已有 Miniconda 环境中的 PyTorch、torchvision 和 Streamlit。
- 为本项目补装的软件包放在 `.venv` 中。
- 基础页面所需的补充依赖见 `requirements-local-ui.txt`。

启动脚本会设置当前进程所需的 DLL 路径及 UTF-8 编码，并暂时关闭 Hugging Face 自动下载。后续准备模型时再单独开启下载。

## 2. 启动后台

在项目目录打开 PowerShell，执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_local_api.ps1
```

后台负责接收请求、调用检测器并保存结果。

访问 `http://127.0.0.1:8000/health`，预期返回：

```json
{"status":"ok"}
```

这个结果只说明后台成功启动。

## 3. 启动操作页面

保持后台运行，再打开一个 PowerShell 窗口，执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_local_dashboard.ps1
```

访问 `http://127.0.0.1:8501`。

页面通过 `http://127.0.0.1:8000/api/v1` 请求后台。关闭服务时，在各自的终端按 `Ctrl+C`。

## 4. 页面启动后的顺序

1. 查看后台连接和检测分支状态。
2. 核对图片检测分支所需的服务或模型。
3. 用已知来源的一张图片验证实际检测流程。
4. 准备真实图片与生成图片，分别记录检测标签和生成模型标签。
5. 训练、评估图片来源分类器，再接入页面。

页面能够打开、检测器成功运行和模型判断准确，需要分别验证。当前的启动脚本不会训练模型或下载数据。

## 5. 图片实验目前已接通

当前使用 `config.local.yaml` 启用本地图片实验模式，原服务器配置仍保留在 `config.yaml`。已准备 CLIP 特征模型和服务器上的图片来源分类器，图片分析不调用云端检测服务。

两张已知真实照片的试运行暴露了误报，当前正式结论保持“不确定”。页面上的实验分数不能解释为 AI 概率。操作方法、原始结果和后续评估步骤见 [本地图片实验说明](LOCAL_IMAGE_EXPERIMENT.md)。

源码中保留着服务器尚未提交的修改，开发时需要区分这些继承的修改与自己的新增修改。
