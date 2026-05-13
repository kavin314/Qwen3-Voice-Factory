@echo off
title Qwen3 Voice Factory

echo Checking for processes using port 7880...

:: 尋找佔用 7880 埠的 PID
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :7880 ^| findstr LISTENING') do (
    set PID=%%a
)

:: 如果找到 PID，則執行刪除動作
if defined PID (
    echo Found process %PID% using port 7880. Killing it...
    taskkill /f /pid %PID%
    timeout /t 2 >nul
) else (
    echo Port 7880 is free.
)


echo Starting App...
set PATH=%PATH%;D:\Software\sox
if not exist "python_env\python.exe" (
    echo ERROR: Python environment not found!
    echo Please run 'install.bat' first.
    pause
    exit
)
set GRADIO_MCP_SERVER=True
set CUDA_LAUNCH_BLOCKING=1
:: 停用 bfloat16 Tensor Op 精度縮減（Pascal 沒有 Tensor Core，啟用時會 CUBLAS_STATUS_EXECUTION_FAILED）
set TORCH_ALLOW_BF16_REDUCED_PRECISION_REDUCTION=0
:: 停用 TF32（Pascal 不支援，開啟無效但保險起見關掉）
set NVIDIA_TF32_OVERRIDE=0

.\python_env\python.exe app.py
pause