# 🎬 Video Dubbing Pipeline

将任意英文视频自动翻译并生成中文配音，产出带双语字幕的最终视频。

**技术栈**：YouTube json3 词级字幕 → NLTK punkt 句子切分 → DeepSeek 翻译 → ElevenLabs TTS → pydub 混音 → ffmpeg 渲染

---

## ✨ 功能

| 步骤 | 功能 |
|------|------|
| 📥 字幕导入 | 支持 YouTube json3（词级时间戳，零重叠）或手动 JSON |
| ✂️ 断句合并 | NLTK punkt + bisect 词级对齐（参考 WhisperX 方案） |
| 🌐 翻译 | DeepSeek API 批量翻译，带上下文参考 |
| 🔊 TTS | ElevenLabs 多语言 v2，动态 speed 参数适配时间槽 |
| 🎚 混音 | pydub overlay，超长音频自动 atempo 压缩 |
| 📝 字幕 | 双语 SRT 生成 |
| 🎬 渲染 | ffmpeg 硬件加速编码（NVIDIA/Mac/Intel/AMD/CPU） |

---

## 🚀 快速开始

### 前置依赖

**ffmpeg**（必须）：
```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg

# Windows
# 下载 https://github.com/BtbN/FFmpeg-Builds/releases
# 解压后将 bin/ 加入 PATH，或放到项目 ffmpeg-btbn/ 目录下
```

**Python 3.10+**

---

### 安装

```bash
git clone https://github.com/your-username/video-dubbing-pipeline.git
cd video-dubbing-pipeline
```

**macOS / Linux**：
```bash
chmod +x run.sh
./run.sh          # 首次运行自动创建 venv 并安装依赖
```

**Windows**：
```
双击 run.bat      # 首次运行自动创建 venv 并安装依赖
```

或手动：
```bash
python -m venv venv
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt_tab')"
python app.py
```

---

### API 密钥配置

在 Gradio 界面的 ⚙️ 配置页填入，或创建 `.env` 文件（程序启动时自动读取）：

```bash
cp .env.example .env
# 编辑 .env，填入你的密钥
```

| 服务 | 用途 | 获取地址 |
|------|------|---------|
| ElevenLabs | TTS 语音合成 | https://elevenlabs.io |
| DeepSeek | 中文翻译 | https://platform.deepseek.com |

---

## 📁 目录结构

```
video-dubbing-pipeline/
├── app.py                  # 主程序（Gradio UI）
├── requirements.txt        # Python 依赖
├── run.bat                 # Windows 启动脚本
├── run.sh                  # macOS / Linux 启动脚本
├── .env.example            # API 密钥模板
├── .gitignore
├── README.md
└── workspace/              # 运行时数据（已 gitignore）
    ├── original_video.mp4          # 源视频（自行放入）
    ├── original_video.en-orig.json3 # YouTube json3 字幕
    ├── translation.json            # 英文分段
    ├── translation_v2.json         # 含翻译的分段
    ├── tts/                        # TTS 音频文件
    ├── saves/                      # 翻译存档
    ├── dubbed_audio_v2.wav         # 混音结果
    ├── subtitles_bilingual.srt     # 双语字幕
    └── dubbed_final_v2.mp4         # 最终视频
```

---

## 🔄 完整工作流

```
1. 下载视频和 json3 字幕
   yt-dlp --write-auto-sub --sub-format json3 --sub-lang en \
          -f "bestvideo[ext=mp4]+bestaudio" -o "workspace/original_video.%(ext)s" \
          "https://www.youtube.com/watch?v=VIDEO_ID"

2. 打开 http://localhost:7860

3. 字幕 → ① 加载字幕 → 📥 从 json3 导入
   （自动用词级时间戳切分句子，无重叠）

4. 字幕 → ② 断句合并（可选，json3 导入后一般不需要）

5. 字幕 → ③ 翻译（DeepSeek）

6. TTS → 生成全部 TTS

7. 混音 → 开始混音

8. 字幕文件 → 生成双语 SRT

9. 渲染 → 开始渲染
```

---

## ⚙️ 参数说明

### TTS 动态语速（语速校准）

程序根据中文字数和时间槽自动计算 ElevenLabs `speed` 参数（范围 0.7~1.2）：

```
speed = min(max(中文字数 / 语速校准 / 时间槽长度, 0.7), 1.2)
```

- **语速校准（字/秒）**：默认 4.5，正常中文朗读约 4~5 字/秒
- 超过 1.2 的部分由混音阶段的 `atempo` 滤镜兜底

### 编码器选择

| 编码器 | 平台 | 说明 |
|--------|------|------|
| h264_videotoolbox (Mac) | macOS | Apple 硬件加速，推荐 |
| h264_nvenc (NVIDIA) | Windows/Linux | NVIDIA GPU，速度最快 |
| libx264 (CPU) | 全平台 | 兼容性最好，速度较慢 |

---

## 🛠 常见问题

**Q: 找不到 ffmpeg？**  
A: 参考上方安装步骤，macOS 用 `brew install ffmpeg`。

**Q: NLTK punkt 下载失败？**  
```bash
python -c "import nltk; nltk.download('punkt_tab')"
```

**Q: TTS 报 quota 不足？**  
A: ElevenLabs 免费额度有限，检查账户用量。

**Q: 渲染字幕乱码？**  
A: 确保字体 `Microsoft YaHei`（Windows）或 `PingFang SC`（macOS）已安装。

---

## 📄 License

MIT
