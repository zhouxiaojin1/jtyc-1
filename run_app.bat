@echo off
echo ========================================
echo 交通流量预测系统 - 启动脚本
echo ========================================
echo.

REM 激活conda环境
echo [1/2] 激活conda环境...
call conda activate ap
if errorlevel 1 (
    echo 错误：无法激活conda环境 'ap'
    echo 请确保已创建该环境
    pause
    exit /b 1
)

echo [2/2] 启动Streamlit应用...
echo.
echo 应用将在浏览器中自动打开
echo 如果没有自动打开，请访问: http://localhost:8501
echo.
echo 按 Ctrl+C 停止应用
echo ========================================
echo.

REM 使用当前环境的Python来运行streamlit
python -m streamlit run app.py

pause
