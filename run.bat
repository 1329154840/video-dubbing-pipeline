@echo off
chcp 65001 >nul
cd /d "%~dp0"

:: 首次运行：自动创建 venv 并安装依赖
if not exist "venv\Scripts\python.exe" (
    echo 🔧 创建虚拟环境...
    python -m venv venv
    echo 📦 安装依赖...
    venv\Scripts\pip.exe install --upgrade pip -q
    venv\Scripts\pip.exe install -r requirements.txt -q
    echo 📥 下载 NLTK 数据...
    venv\Scripts\python.exe -c "import nltk; nltk.download('punkt_tab', quiet=True)"
    echo ✅ 环境初始化完成
)

echo 🚀 启动视频配音 Pipeline...
call venv\Scripts\activate.bat
python app.py
pause
