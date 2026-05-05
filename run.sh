#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

# ── 首次运行：自动创建 venv 并安装依赖 ─────────────────────────────────────────
if [ ! -d "venv" ]; then
    echo "🔧 创建虚拟环境..."
    python3 -m venv venv
    echo "📦 安装依赖..."
    venv/bin/pip install --upgrade pip -q
    venv/bin/pip install -r requirements.txt -q
    echo "📥 下载 NLTK punkt 数据..."
    venv/bin/python -c "import nltk; nltk.download('punkt_tab', quiet=True)"
    echo "✅ 环境初始化完成"
fi

# ── 启动 ────────────────────────────────────────────────────────────────────
echo "🚀 启动视频配音 Pipeline..."
source venv/bin/activate
python app.py
