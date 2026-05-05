import gradio as gr
import json, os, sys, time, re, requests, subprocess, bisect, shutil
import nltk
from openai import OpenAI
from pydub import AudioSegment

# ── Paths ─────────────────────────────────────────────────────────────────────
# BASE = 脚本所在目录（无论从哪里启动都正确）
BASE      = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.join(BASE, "workspace")   # 运行时数据统一放这里
TTS_DIR   = os.path.join(WORKSPACE, "tts")
SAVES_DIR = os.path.join(WORKSPACE, "saves")
for _d in (WORKSPACE, TTS_DIR, SAVES_DIR):
    os.makedirs(_d, exist_ok=True)


def _find_ffmpeg() -> str:
    """
    跨平台查找 ffmpeg：
    1. Windows：优先项目内捆绑的 ffmpeg-btbn/
    2. 系统 PATH（macOS brew install ffmpeg / Linux apt install ffmpeg）
    3. 找不到则给出清晰的安装提示
    """
    if sys.platform == "win32":
        bundled = os.path.join(BASE, "ffmpeg-btbn",
                               "ffmpeg-master-latest-win64-gpl", "bin", "ffmpeg.exe")
        if os.path.exists(bundled):
            return bundled
    found = shutil.which("ffmpeg")
    if found:
        return found
    raise EnvironmentError(
        "找不到 ffmpeg，请先安装：\n"
        "  macOS : brew install ffmpeg\n"
        "  Ubuntu: sudo apt install ffmpeg\n"
        "  Windows: https://github.com/BtbN/FFmpeg-Builds/releases\n"
        "           解压后将 bin/ 目录加入 PATH 或放到项目 ffmpeg-btbn/ 下"
    )


FFMPEG  = _find_ffmpeg()
_ff_dir = os.path.dirname(FFMPEG)
_ff_exe = os.path.basename(FFMPEG)
FFPROBE = os.path.join(_ff_dir, _ff_exe.replace("ffmpeg", "ffprobe"))

os.environ["PATH"] = _ff_dir + os.pathsep + os.environ.get("PATH", "")
AudioSegment.converter = FFMPEG
AudioSegment.ffmpeg    = FFMPEG
AudioSegment.ffprobe   = FFPROBE


def _srt_filter_path(path: str) -> str:
    """为 ffmpeg subtitles= filter 转义路径（Windows/macOS/Linux 各有差异）。"""
    if sys.platform == "win32":
        return path.replace("\\", "/").replace(":", "\\:")
    return path.replace("'", r"\'").replace(":", r"\:")


def _migrate_legacy():
    """一次性将旧的平铺目录结构迁移到 workspace/（升级兼容）。"""
    for old_name, new_dir in [("tts_v2", TTS_DIR), ("translation_saves", SAVES_DIR)]:
        old = os.path.join(BASE, old_name)
        if os.path.isdir(old) and not os.path.isdir(new_dir):
            shutil.move(old, new_dir)
    for fname in ["translation.json", "translation_v2.json",
                  "dubbed_audio_v2.wav", "dubbed_final_v2.mp4",
                  "subtitles_bilingual.srt"]:
        old = os.path.join(BASE, fname)
        new = os.path.join(WORKSPACE, fname)
        if os.path.isfile(old) and not os.path.isfile(new):
            shutil.move(old, new)

_migrate_legacy()


# ── Global state ──────────────────────────────────────────────────────────────
segments = []
original_segments = []   # 保留原始 Whisper 分段，合并后可还原


# ── 存档 / 读档 ────────────────────────────────────────────────────────────────
def list_saves():
    files = sorted(
        [f for f in os.listdir(SAVES_DIR) if f.endswith(".json")],
        reverse=True,
    )
    return files if files else ["（无存档）"]


def save_translation(save_name):
    if not segments:
        return "❌ 无数据可保存", gr.Dropdown(choices=list_saves())
    name = save_name.strip() or time.strftime("%Y%m%d_%H%M%S")
    if not name.endswith(".json"):
        name += ".json"
    path = os.path.join(SAVES_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    return f"✅ 已保存：{name}（{len(segments)} 条）", gr.Dropdown(choices=list_saves(), value=name)


def load_save(save_name):
    global segments
    if not save_name or save_name == "（无存档）":
        return "❌ 请选择存档", [], []
    path = os.path.join(SAVES_DIR, save_name)
    if not os.path.exists(path):
        return f"❌ 找不到 {save_name}", [], []
    with open(path, "r", encoding="utf-8") as f:
        segments = json.load(f)
    return f"✅ 已加载存档：{save_name}（{len(segments)} 条）", segs_to_rows(), segs_to_rows()


def delete_save(save_name):
    if not save_name or save_name == "（无存档）":
        return "❌ 请选择存档", gr.Dropdown(choices=list_saves())
    path = os.path.join(SAVES_DIR, save_name)
    if os.path.exists(path):
        os.remove(path)
        saves = list_saves()
        return f"🗑 已删除：{save_name}", gr.Dropdown(choices=saves, value=saves[0] if saves else None)
    return f"❌ 找不到 {save_name}", gr.Dropdown(choices=list_saves())


# ── 断句合并 (NLTK punkt，参考 WhisperX alignment.py) ─────────────────────────
_NOISE_RE = re.compile(r'^\s*(\[[\w\s]+\]|>>)\s*$')   # 匹配 [music] >> 等噪音

_punkt_splitter = None   # 模块级缓存，避免重复加载

def _load_punkt():
    """
    加载 NLTK punkt 句子分割器（与 WhisperX 一致）。
    优先 punkt_tab（新版），回退到 punkt（旧版）。
    """
    global _punkt_splitter
    if _punkt_splitter is not None:
        return _punkt_splitter
    for pkg in ('punkt_tab', 'punkt'):
        try:
            _punkt_splitter = nltk.data.load(f'tokenizers/{pkg}/english.pickle')
            return _punkt_splitter
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
                _punkt_splitter = nltk.data.load(f'tokenizers/{pkg}/english.pickle')
                return _punkt_splitter
            except Exception:
                continue
    raise RuntimeError("无法加载 NLTK punkt 分词器，请运行: python -m nltk.downloader punkt_tab")


def _dedup_overlapping_segs(raw_segs):
    """
    处理 YouTube / Whisper 滑窗字幕的文本重叠问题：
    1. 跳过零时长纯噪音段（start==end 且文本仅含 [music] >> 等）
    2. 相邻段时间重叠时，找末尾/开头公共词并从后段裁剪
    3. 将后段 start 修正为前段 end，确保无时间重叠
    """
    result = []
    for seg in raw_segs:
        seg = dict(seg)
        # 跳过零时长噪音段
        if seg["start"] >= seg["end"] and _NOISE_RE.match(seg["en"]):
            continue
        if not result:
            result.append(seg)
            continue
        prev = result[-1]
        if seg["start"] < prev["end"]:
            prev_words = prev["en"].split()
            curr_words = seg["en"].split()
            overlap = 0
            for k in range(1, min(len(prev_words), len(curr_words)) + 1):
                if prev_words[-k:] == curr_words[:k]:
                    overlap = k
            if overlap > 0:
                curr_words = curr_words[overlap:]
                seg["en"] = " ".join(curr_words)
            seg["start"] = prev["end"]
        if seg["en"].strip():
            result.append(seg)
    return result


def _split_long_sentence(en_text, start_time, end_time, zh, max_dur, max_chars):
    """将超长句子按词数比例拆为两段，在标点附近寻找切分点。"""
    words = en_text.split()
    if len(words) <= 3:
        return [{"start": start_time, "end": end_time, "en": en_text, "zh": zh}]

    mid = len(words) // 2
    split_at = mid
    # 在 mid 附近找标点
    for offset in range(1, mid + 1):
        for pos in (mid - offset, mid + offset - 1):
            if 0 < pos < len(words) and words[pos].rstrip().endswith((",", ";", "—", "-", "?", "!")):
                split_at = pos + 1
                break
        else:
            continue
        break

    mid_time = start_time + (end_time - start_time) * split_at / len(words)
    part1 = " ".join(words[:split_at])
    part2 = " ".join(words[split_at:])
    result = []
    if part1.strip():
        result.append({"start": start_time, "end": mid_time, "en": part1, "zh": ""})
    if part2.strip():
        result.append({"start": mid_time, "end": end_time, "en": part2, "zh": zh})
    return result


def merge_by_sentence(raw_segs, max_duration=8.0, max_chars=120):
    """
    用 NLTK punkt 检测英文句子边界，合并 Whisper/YouTube 短片段。

    改造自 WhisperX alignment.py 的核心思路：
    ★ 每个词分配独立时间戳（segment 时长按词数等比分配）
      → 即使两句话落在同一 segment 内，也能正确拆分时间
    ★ 用 span_tokenize() 取字符偏移 + bisect 映射回词索引
      → 完全绕开 spaCy/punkt 子词分词与我们词数组的不对齐问题

    步骤：
    1. 去除滑窗重叠文本 (_dedup_overlapping_segs)
    2. 将每个 segment 的时长按词数等比分配 → 每词 (text, w_start, w_end, seg_idx)
    3. NLTK punkt span_tokenize → 字符偏移 → bisect → 词索引
    4. sentence_start = words[wi_start].w_start
       sentence_end   = words[wi_end].w_end
    5. 超长句用 _split_long_sentence 拆分
    6. 过滤零时长/噪音，修剪残留重叠
    """
    splitter = _load_punkt()

    # Step 1
    raw_segs = _dedup_overlapping_segs(raw_segs)
    if not raw_segs:
        return []

    # Step 2: 按词数等比分配时间戳（WhisperX 思路）
    # words[i] = (word_str, w_start_sec, w_end_sec, seg_idx)
    words = []
    tok_char_start = []   # 每词在 full_text 中的字符起始位置（用于 bisect）
    full_text = ""
    for seg_idx, seg in enumerate(raw_segs):
        seg_words = seg["en"].strip().split()
        if not seg_words:
            continue
        n = len(seg_words)
        t0 = seg["start"]
        t1 = seg["end"]
        dur = max(t1 - t0, 0.0)
        for i, w in enumerate(seg_words):
            tok_char_start.append(len(full_text))
            full_text += w + " "
            w_start = t0 + (i / n) * dur
            w_end   = t0 + ((i + 1) / n) * dur
            words.append((w, w_start, w_end, seg_idx))

    if not words:
        return []

    # Step 3: punkt span_tokenize → 字符偏移范围
    sentence_spans = list(splitter.span_tokenize(full_text.strip()))

    def char_to_word_idx(char_pos: int) -> int:
        idx = bisect.bisect_right(tok_char_start, char_pos) - 1
        return max(0, min(idx, len(words) - 1))

    # Step 4 & 5
    merged = []
    for sstart, send in sentence_spans:
        wi_start = char_to_word_idx(sstart)
        wi_end   = char_to_word_idx(send - 1)

        en_text    = " ".join(words[i][0] for i in range(wi_start, wi_end + 1)).strip()
        start_time = words[wi_start][1]   # ★ 词级时间戳，而非整段时间
        end_time   = words[wi_end][2]     # ★ 词级时间戳
        seg_end_idx = words[wi_end][3]
        zh         = raw_segs[seg_end_idx].get("zh", "")
        duration   = end_time - start_time

        if not en_text or _NOISE_RE.match(en_text):
            continue
        if end_time <= start_time:
            continue

        if duration > max_duration or len(en_text) > max_chars:
            merged.extend(
                _split_long_sentence(en_text, start_time, end_time, zh, max_duration, max_chars)
            )
        else:
            merged.append({"start": start_time, "end": end_time, "en": en_text, "zh": zh})

    # Step 6
    merged = [
        s for s in merged
        if s["end"] > s["start"] and s["en"].strip() and not _NOISE_RE.match(s["en"])
    ]
    for i in range(1, len(merged)):
        if merged[i]["start"] < merged[i - 1]["end"]:
            merged[i]["start"] = merged[i - 1]["end"]

    return merged


def run_merge(max_dur, max_chars):
    """Gradio 回调：执行断句合并。"""
    global segments, original_segments
    if not segments:
        return "❌ 请先加载字幕", [], ""

    try:
        original_segments = [s.copy() for s in segments]   # 备份
        merged = merge_by_sentence(segments, float(max_dur), int(max_chars))
        segments = merged

        # 统计
        orig_n   = len(original_segments)
        merged_n = len(merged)
        durs     = [s["end"] - s["start"] for s in merged]
        avg_dur  = sum(durs) / len(durs)
        preview  = "\n".join(
            f"[{s['start']:.1f}s→{s['end']:.1f}s | {s['end']-s['start']:.1f}s]  {s['en']}"
            for s in merged[:20]
        )
        status = (
            f"✅ 合并完成：{orig_n} 段 → {merged_n} 段  "
            f"（平均时长 {avg_dur:.1f}s，前20条预览如下）"
        )
        return status, segs_to_rows(), preview

    except Exception as e:
        return f"❌ 合并失败：{e}", segs_to_rows(), ""


def restore_original():
    """还原为 Whisper 原始分段。"""
    global segments, original_segments
    if not original_segments:
        return "❌ 无备份，请先执行合并", segs_to_rows(), ""
    segments = [s.copy() for s in original_segments]
    return f"✅ 已还原，共 {len(segments)} 段", segs_to_rows(), ""


def fmt_time(s):
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    ms = int(round((s % 1) * 1000))
    return f"{h:02d}:{m:02d}:{int(s % 60):02d},{ms:03d}"


def segs_to_rows():
    return [
        [i, f"{s['start']:.2f}", f"{s['end']:.2f}", s.get("en", ""), s.get("zh", "")]
        for i, s in enumerate(segments)
    ]


def tts_status_rows():
    if not segments:
        return []
    rows = []
    for i, s in enumerate(segments):
        path = os.path.join(TTS_DIR, f"seg_{i:04d}.mp3")
        ok = os.path.exists(path) and os.path.getsize(path) > 500
        rows.append([i, "✅" if ok else "❌", s.get("zh", "")[:60]])
    return rows


# ── Step 1: Load transcript ───────────────────────────────────────────────────
def load_transcript(json_path):
    global segments
    path = json_path.strip() or os.path.join(WORKSPACE, "translation.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            segments = json.load(f)
        v2 = os.path.join(WORKSPACE, "translation_v2.json")
        if os.path.exists(v2):
            with open(v2, "r", encoding="utf-8") as f:
                v2_data = json.load(f)
            for i, seg in enumerate(segments):
                if i < len(v2_data) and v2_data[i].get("zh"):
                    seg["zh"] = v2_data[i]["zh"]
        return f"✅ 已加载 {len(segments)} 条字幕（v2缓存：{'已合并' if os.path.exists(v2) else '无'}）", segs_to_rows()
    except Exception as e:
        return f"❌ {e}", []


# ── json3 词级时间戳导入 ──────────────────────────────────────────────────────
_NOISE_TOKEN_RE = re.compile(r'^\[[\w\s]+\]$|^>>$')   # 单词级噪音过滤


def parse_json3_to_segments(json3_path: str) -> list:
    """
    解析 YouTube json3 字幕文件，利用词级时间戳构建无重叠句子片段。

    改造自 WhisperX alignment.py：
    ★ 每词有独立的 (start_ms, end_ms)
      - start_ms = tStartMs + tOffsetMs（原始 json3 时间）
      - end_ms   = 同一 event 内下一 seg 的开始时间；
                   event 末尾词用 tStartMs + dDurationMs（事件实际结束）
    ★ span_tokenize → bisect → 词索引（同 WhisperX 字符偏移方案）
    ★ sentence_start = words[wi_start].start_ms
      sentence_end   = words[wi_end].end_ms  （真实词尾，非下句开头）
    """
    splitter = _load_punkt()

    with open(json3_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 提取词 + (start_ms, end_ms)，过滤噪音
    # tokens[i] = (word_str, start_ms, end_ms)
    tokens = []
    for ev in data.get("events", []):
        ev_segs = ev.get("segs")
        if not ev_segs:
            continue
        t_start  = ev.get("tStartMs", 0)
        ev_dur   = ev.get("dDurationMs", 0)
        ev_end   = t_start + ev_dur

        for j, seg in enumerate(ev_segs):
            raw    = seg.get("utf8", "")
            s_ms   = t_start + seg.get("tOffsetMs", 0)
            # 词尾 = 同 event 内下一 seg 的起始；event 最后一词用事件结束时间
            if j + 1 < len(ev_segs):
                e_ms = t_start + ev_segs[j + 1].get("tOffsetMs", 0)
            else:
                e_ms = ev_end if ev_dur > 0 else s_ms + 300

            for word in raw.split():
                if _NOISE_TOKEN_RE.match(word):
                    continue
                tokens.append((word, s_ms, e_ms))

    if not tokens:
        return []

    # 构建全文 + 字符偏移表
    full_text = ""
    tok_char_start = []
    for word, _, _ in tokens:
        tok_char_start.append(len(full_text))
        full_text += word + " "

    sentence_spans = list(splitter.span_tokenize(full_text.strip()))

    def char_to_word_idx(char_pos: int) -> int:
        idx = bisect.bisect_right(tok_char_start, char_pos) - 1
        return max(0, min(idx, len(tokens) - 1))

    segs = []
    for sstart, send in sentence_spans:
        wi_start = char_to_word_idx(sstart)
        wi_end   = char_to_word_idx(send - 1)

        en_text  = " ".join(tokens[i][0] for i in range(wi_start, wi_end + 1)).strip()
        if not en_text or _NOISE_RE.match(en_text):
            continue

        start_ms = tokens[wi_start][1]   # ★ 词的真实开始时间
        end_ms   = tokens[wi_end][2]     # ★ 词的真实结束时间（dDurationMs 推算）

        if end_ms <= start_ms:
            end_ms = start_ms + 300

        segs.append({
            "start": round(start_ms / 1000, 3),
            "end":   round(end_ms   / 1000, 3),
            "en":    en_text,
            "zh":    "",
        })

    # 修剪偶发残留重叠（json3 几乎不会出现，但保险）
    for i in range(1, len(segs)):
        if segs[i]["start"] < segs[i - 1]["end"]:
            segs[i]["start"] = segs[i - 1]["end"]

    return segs


def import_from_json3(json3_path: str):
    """Gradio 回调：从 json3 文件导入词级时间戳字幕，覆盖当前 segments。"""
    global segments
    path = json3_path.strip() or os.path.join(WORKSPACE, "original_video.en-orig.json3")
    if not os.path.exists(path):
        return f"❌ 找不到文件：{path}", []
    try:
        segs = parse_json3_to_segments(path)
        if not segs:
            return "❌ 解析结果为空，请检查 json3 文件格式", []
        segments = segs
        # 同步写入 translation.json，作为后续流程的数据源
        out_path = os.path.join(WORKSPACE, "translation.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)
        # 统计
        durs = [s["end"] - s["start"] for s in segs]
        avg  = sum(durs) / len(durs)
        return (
            f"✅ json3 导入完成：{len(segs)} 段，平均时长 {avg:.1f}s，"
            f"已保存 translation.json",
            segs_to_rows(),
        )
    except Exception as e:
        import traceback
        return f"❌ 导入失败：{e}\n{traceback.format_exc()}", []


def save_df_edits(df):
    global segments
    if df is None or not segments:
        return "❌ 无数据"
    try:
        import pandas as pd
        rows = df.values.tolist() if hasattr(df, "values") else df
        for row in rows:
            i = int(row[0])
            if i < len(segments):
                segments[i]["zh"] = str(row[4])
        path = os.path.join(WORKSPACE, "translation_v2.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)
        return f"✅ 已保存 {len(segments)} 条到 translation_v2.json"
    except Exception as e:
        return f"❌ {e}"


# ── Step 2: Translation ───────────────────────────────────────────────────────
def run_translation(batch_size, deepseek_key, progress=gr.Progress()):
    global segments
    if not segments:
        yield "❌ 请先加载字幕", []
        return

    client = OpenAI(api_key=deepseek_key.strip(), base_url="https://api.deepseek.com")
    total = len(segments)
    errors = []

    # 术语对照表
    GLOSSARY = (
        "【术语对照表（保持英文原样）】\n"
        "LLM, AI, Claude, agent, API, MCP, prompt, token, RAG, GPT, "
        "context window, tool use, function calling, workflow, pipeline, "
        "refactor, embedding, fine-tuning, inference, benchmark, hallucination, "
        "memory, retrieval, reasoning, multimodal, latency, throughput"
    )

    # 视频主题背景
    SYSTEM_PROMPT = (
        "你是专业的技术视频字幕翻译员，正在为一段面向中文开发者的 AI/LLM 技术教程视频翻译配音字幕。\n"
        "目标受众：有编程经验的中文开发者。\n"
        "风格要求：口语化、自然流畅，适合配音朗读，避免书面腔。\n"
        + GLOSSARY
    )

    for start in range(0, total, int(batch_size)):
        end = min(start + int(batch_size), total)
        batch_en = [segments[i]["en"] for i in range(start, end)]
        progress(start / total, desc=f"翻译 {start}/{total}...")

        # 跨批次上下文：提供前 6 条已译结果作为参考
        context_str = ""
        if start > 0:
            ctx_start = max(0, start - 6)
            ctx_lines = [
                f"  [{i}] {segments[i].get('zh', '')}"
                for i in range(ctx_start, start)
                if segments[i].get("zh")
            ]
            if ctx_lines:
                context_str = "【上文已译（仅供风格参考，勿修改）】\n" + "\n".join(ctx_lines) + "\n\n"

        prompt = (
            f"{context_str}"
            f"【待翻译：共 {len(batch_en)} 条英文字幕】\n"
            f"{json.dumps(batch_en, ensure_ascii=False)}\n\n"
            "要求：\n"
            "- 每条译文长度尽量与原文相近，不要过长\n"
            "- 术语对照表中的词汇保持英文原样\n"
            "- 直接返回含 "
            f"{len(batch_en)} 个字符串的 JSON 数组，不加任何说明"
        )

        success = False
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model="deepseek-v4-flash",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": prompt},
                    ],
                    max_tokens=4096,
                    temperature=0.3,
                )
                text = resp.choices[0].message.content.strip()
                match = re.search(r'\[.*\]', text, re.DOTALL)
                if match:
                    batch_zh = json.loads(match.group())
                    if len(batch_zh) == len(batch_en):
                        for i, zh in enumerate(batch_zh):
                            segments[start + i]["zh"] = zh
                        success = True
                        break
            except Exception:
                time.sleep(2)

        if not success:
            errors.append(f"{start}-{end}")

        status = f"翻译进度: {end}/{total}"
        if errors:
            status += f"  ⚠️ 回退批次: {', '.join(errors)}"
        yield status, segs_to_rows()

    path = os.path.join(WORKSPACE, "translation_v2.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    yield f"✅ 翻译完成！{total} 条。失败批次: {errors if errors else '无'}", segs_to_rows()


def reset_translation_cache():
    path = os.path.join(WORKSPACE, "translation_v2.json")
    if os.path.exists(path):
        os.remove(path)
        return "✅ 已删除 translation_v2.json 缓存，可重新翻译"
    return "ℹ️ 缓存不存在"


# ── TTS 动态语速计算 ──────────────────────────────────────────────────────────
def calc_speed(zh_text: str, slot_sec: float, chars_per_sec: float = 4.5) -> float:
    """
    根据中文字符数估算朗读时长，计算 ElevenLabs speed 参数（0.7 ~ 1.2）。
    - slot_sec : 当前时间槽长度（秒）
    - chars_per_sec : 正常语速（字/秒），默认 4.5，可在 UI 校准
    返回值 1.0 表示无需调整；>1.0 表示需要加速（文字多于时间槽）。
    """
    if slot_sec <= 0:
        return 1.0
    n = len(zh_text.replace(" ", ""))
    if n == 0:
        return 1.0
    estimated = n / chars_per_sec          # 自然朗读预估时长
    ratio = estimated / slot_sec           # >1 表示需要加速
    if ratio < 0.95:                       # 留 5% 余量，不减速
        return 1.0
    return round(min(max(ratio, 0.7), 1.2), 3)   # 限制在 ElevenLabs 支持范围


# ── Step 3: TTS ───────────────────────────────────────────────────────────────
def run_tts(eleven_key, voice_id, stability, similarity, style, chars_per_sec,
            progress=gr.Progress()):
    if not segments:
        yield "❌ 请先加载字幕", []
        return

    total = len(segments)
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id.strip()}"
    errors = []

    for idx, seg in enumerate(segments):
        out_path = os.path.join(TTS_DIR, f"seg_{idx:04d}.mp3")
        if os.path.exists(out_path) and os.path.getsize(out_path) > 500:
            progress(idx / total, desc=f"跳过 {idx}/{total}")
            continue

        progress(idx / total, desc=f"TTS {idx + 1}/{total}")
        slot_sec = seg.get("end", 0) - seg.get("start", 0)
        spd = calc_speed(seg.get("zh", ""), slot_sec, float(chars_per_sec))
        for attempt in range(3):
            try:
                r = requests.post(
                    url,
                    headers={"xi-api-key": eleven_key.strip(), "Content-Type": "application/json"},
                    json={
                        "text": seg.get("zh", ""),
                        "model_id": "eleven_multilingual_v2",
                        "voice_settings": {
                            "stability": stability,
                            "similarity_boost": similarity,
                            "style": style,
                            "speed": spd,
                            "use_speaker_boost": True,
                        },
                    },
                    timeout=30,
                )
                if r.status_code == 200:
                    with open(out_path, "wb") as f:
                        f.write(r.content)
                    break
                elif r.status_code == 429:
                    time.sleep(12 * (attempt + 1))
                else:
                    err = r.json().get("detail", {})
                    if "quota" in str(err):
                        yield f"❌ 配额不足，已停止 ({idx}/{total})", tts_status_rows()
                        return
                    break
            except Exception:
                time.sleep(3)
        else:
            errors.append(idx)

        time.sleep(0.15)
        if (idx + 1) % 20 == 0 or idx == total - 1:
            done = sum(1 for fn in os.listdir(TTS_DIR) if fn.endswith(".mp3"))
            yield f"TTS进度: {done}/{total}  错误: {len(errors)}", tts_status_rows()

    done = sum(1 for fn in os.listdir(TTS_DIR) if fn.endswith(".mp3"))
    yield f"✅ TTS完成！{done}/{total}  失败段: {errors if errors else '无'}", tts_status_rows()


def play_segment(seg_idx):
    idx = int(seg_idx)
    if idx >= len(segments):
        return None, "❌ 索引超出范围"
    path = os.path.join(TTS_DIR, f"seg_{idx:04d}.mp3")
    zh = segments[idx].get("zh", "")
    en = segments[idx].get("en", "")
    info = f"[{idx}] {segments[idx]['start']:.2f}s → {segments[idx]['end']:.2f}s\n中：{zh}\n英：{en}"
    if os.path.exists(path) and os.path.getsize(path) > 500:
        return path, info
    return None, f"❌ 音频不存在\n{info}"


def regenerate_segment(seg_idx, eleven_key, voice_id, stability, similarity, style, chars_per_sec):
    idx = int(seg_idx)
    if idx >= len(segments):
        return "❌ 无效索引", None
    out_path = os.path.join(TTS_DIR, f"seg_{idx:04d}.mp3")
    if os.path.exists(out_path):
        os.remove(out_path)
    spd_path = out_path.replace(".mp3", "_spd.mp3")
    if os.path.exists(spd_path):
        os.remove(spd_path)

    seg = segments[idx]
    slot_sec = seg.get("end", 0) - seg.get("start", 0)
    spd = calc_speed(seg.get("zh", ""), slot_sec, float(chars_per_sec))

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id.strip()}"
    for attempt in range(3):
        try:
            r = requests.post(
                url,
                headers={"xi-api-key": eleven_key.strip(), "Content-Type": "application/json"},
                json={
                    "text": seg.get("zh", ""),
                    "model_id": "eleven_multilingual_v2",
                    "voice_settings": {
                        "stability": stability,
                        "similarity_boost": similarity,
                        "style": style,
                        "speed": spd,
                        "use_speaker_boost": True,
                    },
                },
                timeout=30,
            )
            if r.status_code == 200:
                with open(out_path, "wb") as f:
                    f.write(r.content)
                return f"✅ seg_{idx:04d} 重新生成成功", out_path
        except Exception:
            time.sleep(2)
    return f"❌ seg_{idx:04d} 生成失败", None


def refresh_tts_status():
    done = sum(1 for fn in os.listdir(TTS_DIR) if fn.endswith(".mp3") and not fn.endswith("_spd.mp3"))
    total = len(segments)
    return f"已完成: {done}/{total}", tts_status_rows()


def regenerate_range(start_idx, end_idx, eleven_key, voice_id, stability, similarity, style,
                     chars_per_sec, progress=gr.Progress()):
    """批量重新生成指定范围的 TTS 段落。"""
    if not segments:
        yield "❌ 请先加载字幕", tts_status_rows()
        return
    s = int(start_idx)
    e = min(int(end_idx), len(segments) - 1)
    if s > e:
        yield f"❌ 范围无效：{s} > {e}", tts_status_rows()
        return

    # 删除该范围内的旧文件
    for i in range(s, e + 1):
        for suffix in ["", "_spd"]:
            p = os.path.join(TTS_DIR, f"seg_{i:04d}{suffix}.mp3")
            if os.path.exists(p):
                os.remove(p)

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id.strip()}"
    errors = []
    total_range = e - s + 1

    for i, idx in enumerate(range(s, e + 1)):
        progress(i / total_range, desc=f"重新生成 {idx}/{e}")
        out_path = os.path.join(TTS_DIR, f"seg_{idx:04d}.mp3")
        seg = segments[idx]
        slot_sec = seg.get("end", 0) - seg.get("start", 0)
        spd = calc_speed(seg.get("zh", ""), slot_sec, float(chars_per_sec))
        ok = False
        for attempt in range(3):
            try:
                r = requests.post(
                    url,
                    headers={"xi-api-key": eleven_key.strip(), "Content-Type": "application/json"},
                    json={
                        "text": seg.get("zh", ""),
                        "model_id": "eleven_multilingual_v2",
                        "voice_settings": {
                            "stability": stability, "similarity_boost": similarity,
                            "style": style, "speed": spd, "use_speaker_boost": True,
                        },
                    },
                    timeout=30,
                )
                if r.status_code == 200:
                    with open(out_path, "wb") as f:
                        f.write(r.content)
                    ok = True
                    break
                elif r.status_code == 429:
                    time.sleep(12 * (attempt + 1))
                else:
                    err = r.json().get("detail", {})
                    if "quota" in str(err):
                        yield f"❌ 配额不足，已停止（{idx}）", tts_status_rows()
                        return
                    break
            except Exception:
                time.sleep(3)
        if not ok:
            errors.append(idx)
        time.sleep(0.15)

        if (i + 1) % 10 == 0 or idx == e:
            yield (f"批量重生成进度: {i+1}/{total_range}  失败: {len(errors)}",
                   tts_status_rows())

    progress(1.0)
    yield (f"✅ 批量重生成完成！范围 {s}-{e}，失败: {errors if errors else '无'}",
           tts_status_rows())


def clear_all_tts():
    """删除全部 TTS 音频文件，准备完整重新生成。"""
    files = [f for f in os.listdir(TTS_DIR) if f.endswith(".mp3")]
    for fn in files:
        os.remove(os.path.join(TTS_DIR, fn))
    return f"🗑 已清除 {len(files)} 个音频文件，可重新生成", tts_status_rows()


# ── Step 4: Mix ───────────────────────────────────────────────────────────────
def run_mix(max_speed, progress=gr.Progress()):
    if not segments:
        yield "❌ 请先加载字幕", None
        return

    total = len(segments)
    total_ms = int((segments[-1]["end"] + 1.5) * 1000)
    progress(0, desc="初始化音轨...")
    dubbed = AudioSegment.silent(duration=total_ms, frame_rate=44100)
    mix_errors = 0

    for idx, seg in enumerate(segments):
        out_path = os.path.join(TTS_DIR, f"seg_{idx:04d}.mp3")
        if not os.path.exists(out_path) or os.path.getsize(out_path) < 500:
            mix_errors += 1
            continue
        try:
            audio = AudioSegment.from_mp3(out_path)
        except Exception:
            mix_errors += 1
            continue

        start_ms = int(seg["start"] * 1000)
        slot_ms  = int((seg["end"] - seg["start"]) * 1000)
        audio_ms = len(audio)

        if audio_ms > slot_ms * max_speed and slot_ms > 300:
            speed = min(audio_ms / slot_ms, max_speed)
            tmp = out_path.replace(".mp3", "_spd.mp3")
            if not os.path.exists(tmp):
                subprocess.run(
                    [FFMPEG, "-y", "-i", out_path, "-filter:a", f"atempo={speed:.3f}", tmp],
                    capture_output=True,
                )
            if os.path.exists(tmp):
                try:
                    audio = AudioSegment.from_mp3(tmp)
                except Exception:
                    pass

        dubbed = dubbed.overlay(audio, position=start_ms)

        if (idx + 1) % 50 == 0:
            progress((idx + 1) / total, desc=f"混音 {idx + 1}/{total}")
            yield f"混音进度: {idx + 1}/{total}  跳过: {mix_errors}", None

    wav_path = os.path.join(WORKSPACE, "dubbed_audio_v2.wav")
    progress(0.95, desc="导出 WAV...")
    dubbed.export(wav_path, format="wav")
    rms = AudioSegment.from_wav(wav_path)[:5000].rms
    label = "OK" if rms > 10 else "WARNING: 音频可能为空"
    yield f"✅ 混音完成！RMS={rms} ({label})  跳过: {mix_errors}", wav_path


# ── Step 5: SRT ───────────────────────────────────────────────────────────────
def generate_srt():
    if not segments:
        return "❌ 请先加载字幕", ""
    srt_path = os.path.join(WORKSPACE, "subtitles_bilingual.srt")
    lines = []
    for i, seg in enumerate(segments):
        lines += [
            str(i + 1),
            f"{fmt_time(seg['start'])} --> {fmt_time(seg['end'])}",
            seg.get("zh", ""),
            seg.get("en", ""),
            "",
        ]
    content = "\n".join(lines)
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(content)
    preview = "\n".join(lines[:60])
    return f"✅ 已生成 subtitles_bilingual.srt（{len(segments)} 条）", preview


# ── Step 6: Render ────────────────────────────────────────────────────────────
def _get_duration(path):
    """用 ffprobe 获取视频总时长（秒）。"""
    result = subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True, errors="ignore",
    )
    try:
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def run_render(video_path, font_size, crf, encoder, progress=gr.Progress()):
    wav_path    = os.path.join(WORKSPACE, "dubbed_audio_v2.wav")
    srt_path    = os.path.join(WORKSPACE, "subtitles_bilingual.srt")
    final_video = os.path.join(WORKSPACE, "dubbed_final_v2.mp4")

    if not os.path.exists(wav_path):
        yield "❌ 找不到 dubbed_audio_v2.wav，请先完成混音", None
        return

    # 渲染前自动用当前内存中的 segments 重新生成最新 SRT
    if segments:
        lines = []
        for i, seg in enumerate(segments):
            lines += [
                str(i + 1),
                f"{fmt_time(seg['start'])} --> {fmt_time(seg['end'])}",
                seg.get("zh", ""),
                seg.get("en", ""),
                "",
            ]
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        yield f"✅ 已自动更新字幕文件（{len(segments)} 条），启动渲染...", None
    elif not os.path.exists(srt_path):
        yield "❌ 找不到字幕文件，请先加载字幕并生成 SRT", None
        return

    # 获取总时长用于计算进度百分比
    total_sec = _get_duration(video_path.strip()) or 1.0
    total_str = time.strftime("%H:%M:%S", time.gmtime(total_sec))

    srt_escaped = _srt_filter_path(srt_path)
    progress(0, desc="启动 ffmpeg...")
    yield f"⏳ 启动渲染，视频总时长 {total_str}...", None

    # 编码器参数
    ENCODER_PARAMS = {
        "libx264 (CPU)":           ["-c:v", "libx264",            "-preset", "fast",     "-crf",             str(int(crf))],
        "h264_videotoolbox (Mac)": ["-c:v", "h264_videotoolbox",  "-q:v",    str(int(crf))],
        "h264_nvenc (NVIDIA)":     ["-c:v", "h264_nvenc",         "-preset", "p4",       "-cq",              str(int(crf)), "-gpu", "0"],
        "hevc_nvenc (NVIDIA)":     ["-c:v", "hevc_nvenc",         "-preset", "p4",       "-cq",              str(int(crf)), "-gpu", "0"],
        "h264_amf (AMD)":          ["-c:v", "h264_amf",           "-quality","balanced", "-qp_i",            str(int(crf))],
        "h264_qsv (Intel)":        ["-c:v", "h264_qsv",           "-preset", "fast",     "-global_quality",  str(int(crf))],
    }
    enc_args = ENCODER_PARAMS.get(encoder, ENCODER_PARAMS["libx264 (CPU)"])

    cmd = [
        FFMPEG, "-y",
        "-i", video_path.strip(),
        "-i", wav_path,
        "-vf", (
            f"subtitles='{srt_escaped}'"
            f":force_style='FontName=Microsoft YaHei,FontSize={int(font_size)},"
            f"PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
            f"Outline=2,Shadow=1,Alignment=2,MarginV=20'"
        ),
        *enc_args,
        "-c:a", "aac", "-b:a", "128k",
        "-map", "0:v:0", "-map", "1:a:0",
        "-progress", "pipe:1",
        "-nostats",
        "-shortest", final_video,
    ]

    import threading

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True, errors="ignore",
    )

    # 用独立线程持续排空 stderr，防止缓冲区满后阻塞主进程
    stderr_lines = []
    def _drain_stderr():
        for line in proc.stderr:
            stderr_lines.append(line)
    t = threading.Thread(target=_drain_stderr, daemon=True)
    t.start()

    current_sec = 0.0
    fps = speed = "—"
    last_yield_pct = -1.0
    last_yield_time = time.time()

    while True:
        line = proc.stdout.readline()
        if not line and proc.poll() is not None:
            break
        line = line.strip()
        if line.startswith("out_time_ms="):
            try:
                val = int(line.split("=")[1])
                if val >= 0:
                    current_sec = val / 1_000_000
            except Exception:
                pass
        elif line.startswith("fps="):
            fps = line.split("=")[1].strip() or fps
        elif line.startswith("speed="):
            speed = line.split("=")[1].strip() or speed
        elif line.startswith("progress="):
            pct = min(current_sec / total_sec, 1.0)
            now = time.time()
            # 只在进度变化 ≥1% 且距上次更新 ≥2 秒时才 yield，减少闪烁
            if pct - last_yield_pct >= 0.01 and now - last_yield_time >= 2.0:
                cur_str = time.strftime("%H:%M:%S", time.gmtime(current_sec))
                desc = f"渲染中 {cur_str} / {total_str}  fps={fps}  speed={speed}"
                progress(pct, desc=desc)
                yield f"⏳ {desc}  ({pct*100:.1f}%)", None
                last_yield_pct = pct
                last_yield_time = now

    t.join(timeout=5)
    proc.wait()

    if proc.returncode == 0:
        size_mb = os.path.getsize(final_video) / 1024 / 1024
        progress(1.0, desc="完成")
        yield f"✅ 渲染完成！{size_mb:.1f} MB → dubbed_final_v2.mp4", final_video
    else:
        err = "".join(stderr_lines[-30:])
        yield f"❌ 渲染失败：\n{err}", None


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="视频配音 Pipeline") as app:
    gr.Markdown(
        "# 🎬 视频配音 Pipeline\n"
        "> DeepSeek 翻译 → ElevenLabs TTS → 双语字幕 → 视频合成"
    )

    with gr.Tabs():

        # ── 配置 ──────────────────────────────────────────────────────────────
        with gr.Tab("⚙️ 配置"):
            gr.Markdown("### API 密钥 & 路径")
            with gr.Row():
                eleven_key_in = gr.Textbox(
                    label="ElevenLabs API Key",
                    value=os.environ.get("ELEVEN_API_KEY", ""),
                    type="password", scale=2,
                )
                deepseek_key_in = gr.Textbox(
                    label="DeepSeek API Key",
                    value=os.environ.get("DEEPSEEK_API_KEY", ""),
                    type="password", scale=2,
                )
            with gr.Row():
                voice_id_in = gr.Textbox(
                    label="Voice ID",
                    value=os.environ.get("ELEVENLABS_VOICE_ID", ""),
                    scale=2,
                )
            gr.Markdown("---\n### 文件路径")
            with gr.Row():
                video_path_in  = gr.Textbox(
                    label="原始视频路径",
                    value=os.path.join(WORKSPACE, "original_video.mp4"), scale=4,
                )
                video_pick_btn = gr.Button("📁 选择", scale=0, min_width=70)
            with gr.Row():
                transcript_path_in  = gr.Textbox(
                    label="字幕 JSON 路径",
                    value=os.path.join(WORKSPACE, "translation.json"), scale=4,
                )
                transcript_pick_btn = gr.Button("📁 选择", scale=0, min_width=70)

            def pick_file(current, filetypes):
                import tkinter as tk
                from tkinter import filedialog
                root = tk.Tk()
                root.withdraw()
                root.wm_attributes("-topmost", 1)
                path = filedialog.askopenfilename(
                    initialdir=os.path.dirname(current) if current and os.path.exists(os.path.dirname(current)) else WORKSPACE,
                    filetypes=filetypes,
                )
                root.destroy()
                return path if path else current

            video_pick_btn.click(
                lambda p: pick_file(p, [("视频文件", "*.mp4 *.mkv *.avi *.mov"), ("所有文件", "*.*")]),
                inputs=[video_path_in], outputs=[video_path_in],
            )
            transcript_pick_btn.click(
                lambda p: pick_file(p, [("JSON 文件", "*.json"), ("所有文件", "*.*")]),
                inputs=[transcript_path_in], outputs=[transcript_path_in],
            )

        # ── 字幕 + 断句合并 + 翻译 + 存档 ───────────────────────────────────────
        with gr.Tab("📄 字幕"):

            # ① 加载
            with gr.Accordion("① 加载字幕", open=True):
                with gr.Row():
                    load_btn      = gr.Button("📂 加载 JSON 字幕", variant="primary")
                    save_edit_btn = gr.Button("💾 保存编辑", variant="secondary")
                transcript_status = gr.Textbox(label="状态", interactive=False)

                gr.Markdown("---\n**📥 从 YouTube json3 导入（词级时间戳，无重叠）**")
                with gr.Row():
                    json3_path_in = gr.Textbox(
                        label="json3 文件路径",
                        value=os.path.join(WORKSPACE, "original_video.en-orig.json3"),
                        scale=4,
                    )
                    import_json3_btn = gr.Button("📥 从 json3 导入", variant="primary", scale=1)
                gr.Markdown(
                    "> 优先使用此方式：YouTube json3 含词级时间戳，"
                    "句子边界精确且**零重叠**。导入后自动覆盖 `translation.json`。"
                )

                seg_df = gr.Dataframe(
                    headers=["#", "开始(s)", "结束(s)", "英文", "中文"],
                    datatype=["number", "str", "str", "str", "str"],
                    col_count=(5, "fixed"),
                    interactive=True,
                    wrap=True,
                    label="字幕列表（中文列可直接编辑）",
                )
                load_btn.click(load_transcript, inputs=[transcript_path_in], outputs=[transcript_status, seg_df])
                save_edit_btn.click(save_df_edits, inputs=[seg_df], outputs=[transcript_status])
                import_json3_btn.click(
                    import_from_json3,
                    inputs=[json3_path_in],
                    outputs=[transcript_status, seg_df],
                )

            # ② 断句合并
            with gr.Accordion("② 断句合并（spaCy）", open=False):
                gr.Markdown("> 将 Whisper 短片段合并为完整句子，**先合并再翻译**效果更好。")
                with gr.Row():
                    max_dur_sl   = gr.Slider(3.0, 15.0, value=8.0, step=0.5,
                                             label="单句最大时长（秒）", scale=3)
                    max_chars_sl = gr.Slider(40, 200, value=120, step=10,
                                             label="单句最大字符数", scale=3)
                with gr.Row():
                    merge_btn   = gr.Button("▶ 执行合并", variant="primary", scale=1)
                    restore_btn = gr.Button("↩ 还原原始分段", variant="secondary", scale=1)
                merge_status  = gr.Textbox(label="状态", interactive=False)
                merge_preview = gr.Textbox(
                    label="前 20 句预览（时长 + 文本）", lines=12, interactive=False)
                merge_df = gr.Dataframe(
                    headers=["#", "开始(s)", "结束(s)", "英文", "中文"],
                    datatype=["number", "str", "str", "str", "str"],
                    col_count=(5, "fixed"), interactive=False, wrap=True,
                    label="合并后字幕",
                )
                merge_btn.click(
                    run_merge,
                    inputs=[max_dur_sl, max_chars_sl],
                    outputs=[merge_status, merge_df, merge_preview],
                )
                restore_btn.click(
                    restore_original,
                    outputs=[merge_status, merge_df, merge_preview],
                )

            # ③ 翻译
            with gr.Accordion("③ 翻译（DeepSeek）", open=False):
                with gr.Row():
                    batch_slider    = gr.Slider(10, 80, value=40, step=10, label="批次大小", scale=3)
                    translate_btn   = gr.Button("▶ 开始翻译", variant="primary", scale=1)
                    reset_trans_btn = gr.Button("🗑 重置缓存", variant="stop", scale=1)
                trans_status = gr.Textbox(label="状态", interactive=False)
                trans_df = gr.Dataframe(
                    headers=["#", "开始(s)", "结束(s)", "英文", "中文"],
                    datatype=["number", "str", "str", "str", "str"],
                    col_count=(5, "fixed"), interactive=True, wrap=True,
                    label="翻译结果（实时更新，可手动编辑）",
                )
                translate_btn.click(
                    run_translation,
                    inputs=[batch_slider, deepseek_key_in],
                    outputs=[trans_status, trans_df],
                )
                reset_trans_btn.click(reset_translation_cache, outputs=[trans_status])

            # ④ 存档
            with gr.Accordion("④ 存档管理", open=False):
                with gr.Row():
                    save_name_in = gr.Textbox(
                        label="存档名称（留空用时间戳）",
                        placeholder="例：deepseek_v4flash_batch40",
                        scale=3,
                    )
                    save_btn = gr.Button("💾 保存当前译文", variant="primary", scale=1)
                save_status = gr.Textbox(label="状态", interactive=False)
                gr.Markdown("---")
                with gr.Row():
                    saves_dd          = gr.Dropdown(choices=list_saves(), label="选择存档", scale=3)
                    refresh_saves_btn = gr.Button("🔄", scale=0, min_width=60)
                with gr.Row():
                    load_save_btn   = gr.Button("📂 加载", variant="primary", scale=1)
                    delete_save_btn = gr.Button("🗑 删除", variant="stop",    scale=1)
                saves_df = gr.Dataframe(
                    headers=["#", "开始(s)", "结束(s)", "英文", "中文"],
                    datatype=["number", "str", "str", "str", "str"],
                    col_count=(5, "fixed"), interactive=False, wrap=True,
                    label="存档内容预览",
                )
                save_btn.click(save_translation, inputs=[save_name_in], outputs=[save_status, saves_dd])
                load_save_btn.click(load_save, inputs=[saves_dd], outputs=[save_status, seg_df, saves_df])
                delete_save_btn.click(delete_save, inputs=[saves_dd], outputs=[save_status, saves_dd])
                refresh_saves_btn.click(lambda: gr.Dropdown(choices=list_saves()), outputs=[saves_dd])

        # ── TTS ───────────────────────────────────────────────────────────────
        with gr.Tab("🔊 TTS"):
            gr.Markdown("### 语音参数")
            with gr.Row():
                stability_sl  = gr.Slider(0, 1, value=0.5, step=0.05, label="Stability")
                similarity_sl = gr.Slider(0, 1, value=0.8, step=0.05, label="Similarity Boost")
                style_sl      = gr.Slider(0, 1, value=0.2, step=0.05, label="Style")
            with gr.Row():
                chars_per_sec_sl = gr.Slider(
                    3.0, 7.0, value=4.5, step=0.1,
                    label="语速校准（字/秒）— 用于估算 TTS speed 参数",
                    info="中文正常语速约 4~5 字/秒。值越大 → 估算越短 → speed 越接近 1.0",
                    scale=3,
                )
            gr.Markdown(
                "> 💡 **动态 speed 说明**：生成前根据中文字数 ÷ 时间槽估算所需语速，"
                "自动设置 ElevenLabs `speed`（范围 0.7~1.2）。超出 1.2 的部分由混音时 `atempo` 兜底。"
            )
            with gr.Row():
                tts_btn         = gr.Button("▶ 生成全部 TTS", variant="primary")
                refresh_tts_btn = gr.Button("🔄 刷新状态", variant="secondary")
            tts_status = gr.Textbox(label="状态", interactive=False)
            tts_df = gr.Dataframe(
                headers=["#", "状态", "中文文本"],
                datatype=["number", "str", "str"],
                col_count=(3, "fixed"),
                interactive=False,
                label="各段生成状态（✅ 已完成 / ❌ 未生成）",
            )
            gr.Markdown("---\n### 🎧 单段预览 & 重新生成")
            with gr.Row():
                seg_idx_in = gr.Number(label="段落编号", value=0, precision=0, scale=1)
                play_btn   = gr.Button("▶ 播放", variant="secondary", scale=1)
                regen_btn  = gr.Button("🔄 重新生成此段", variant="secondary", scale=1)
            seg_info  = gr.Textbox(label="段落信息", interactive=False, lines=3)
            seg_audio = gr.Audio(label="音频预览", type="filepath")

            gr.Markdown("---\n### 🔁 批量重新生成")
            with gr.Row():
                range_start = gr.Number(label="起始段落", value=0,   precision=0, scale=1)
                range_end   = gr.Number(label="结束段落", value=50,  precision=0, scale=1)
                regen_range_btn  = gr.Button("▶ 重新生成此范围", variant="primary",   scale=1)
                clear_all_tts_btn = gr.Button("🗑 清除全部重新生成", variant="stop", scale=1)
            gr.Markdown(
                "> `清除全部` 会删除所有已生成的音频，再点「生成全部 TTS」从头开始。\n"
                "> `重新生成此范围` 只删除指定范围的旧文件，其余不受影响。"
            )

            tts_btn.click(
                run_tts,
                inputs=[eleven_key_in, voice_id_in, stability_sl, similarity_sl, style_sl,
                        chars_per_sec_sl],
                outputs=[tts_status, tts_df],
            )
            refresh_tts_btn.click(refresh_tts_status, outputs=[tts_status, tts_df])
            play_btn.click(play_segment, inputs=[seg_idx_in], outputs=[seg_audio, seg_info])
            regen_btn.click(
                regenerate_segment,
                inputs=[seg_idx_in, eleven_key_in, voice_id_in, stability_sl, similarity_sl,
                        style_sl, chars_per_sec_sl],
                outputs=[seg_info, seg_audio],
            )
            regen_range_btn.click(
                regenerate_range,
                inputs=[range_start, range_end,
                        eleven_key_in, voice_id_in, stability_sl, similarity_sl, style_sl,
                        chars_per_sec_sl],
                outputs=[tts_status, tts_df],
            )
            clear_all_tts_btn.click(clear_all_tts, outputs=[tts_status, tts_df])

        # ── 混音 ──────────────────────────────────────────────────────────────
        with gr.Tab("🎚 混音"):
            max_speed_sl = gr.Slider(1.0, 2.0, value=1.5, step=0.1, label="最大加速倍率（超出则压缩语速）")
            mix_btn      = gr.Button("▶ 开始混音", variant="primary")
            mix_status   = gr.Textbox(label="状态", interactive=False)
            mix_audio    = gr.Audio(label="混音预览", type="filepath")
            mix_btn.click(run_mix, inputs=[max_speed_sl], outputs=[mix_status, mix_audio])

        # ── 字幕文件 ──────────────────────────────────────────────────────────
        with gr.Tab("📝 字幕文件"):
            gen_srt_btn = gr.Button("▶ 生成双语 SRT", variant="primary")
            srt_status  = gr.Textbox(label="状态", interactive=False)
            srt_preview = gr.Textbox(label="SRT 预览（前60行）", lines=20, interactive=False)
            gen_srt_btn.click(generate_srt, outputs=[srt_status, srt_preview])

        # ── 渲染 ──────────────────────────────────────────────────────────────
        with gr.Tab("🎬 渲染"):

            def get_render_files_info():
                def file_info(path):
                    if os.path.exists(path):
                        size_mb = os.path.getsize(path) / 1024 / 1024
                        mtime   = time.strftime("%Y-%m-%d %H:%M:%S",
                                    time.localtime(os.path.getmtime(path)))
                        return f"✅  {path}\n    大小: {size_mb:.1f} MB　更新时间: {mtime}"
                    return f"❌  {path}\n    （文件不存在）"

                srt = os.path.join(WORKSPACE, "subtitles_bilingual.srt")
                wav = os.path.join(WORKSPACE, "dubbed_audio_v2.wav")
                return file_info(srt), file_info(wav)

            with gr.Row():
                srt_info_box = gr.Textbox(
                    label="📝 字幕文件", lines=2, interactive=False, scale=1)
                wav_info_box = gr.Textbox(
                    label="🎚 混音文件", lines=2, interactive=False, scale=1)
            refresh_files_btn = gr.Button("🔄 刷新文件信息", variant="secondary", size="sm")
            refresh_files_btn.click(get_render_files_info, outputs=[srt_info_box, wav_info_box])

            # 页面加载时自动填充
            app.load(get_render_files_info, outputs=[srt_info_box, wav_info_box])

            gr.Markdown("---")
            with gr.Row():
                _default_enc = (
                    "h264_videotoolbox (Mac)" if sys.platform == "darwin"
                    else "h264_nvenc (NVIDIA)"
                )
                encoder_dd   = gr.Dropdown(
                    choices=[
                        "libx264 (CPU)",
                        "h264_videotoolbox (Mac)",
                        "h264_nvenc (NVIDIA)",
                        "hevc_nvenc (NVIDIA)",
                        "h264_amf (AMD)",
                        "h264_qsv (Intel)",
                    ],
                    value=_default_enc,
                    label="编码器",
                    scale=2,
                )
                font_size_sl = gr.Slider(10, 30, value=16, step=1, label="字幕字号", scale=2)
                crf_sl       = gr.Slider(18, 28, value=22, step=1, label="质量 CRF/CQ（越小越清晰）", scale=2)
            gr.Markdown(
                "> 💡 **编码器说明**：Mac → h264_videotoolbox（硬件加速）；"
                "NVIDIA → h264_nvenc；无 GPU → libx264 (CPU)"
            )
            render_btn      = gr.Button("▶ 开始渲染", variant="primary")
            render_status   = gr.Textbox(label="状态", interactive=False)
            final_video_out = gr.Video(label="最终视频预览")

            render_btn.click(
                run_render,
                inputs=[video_path_in, font_size_sl, crf_sl, encoder_dd],
                outputs=[render_status, final_video_out],
            )
            # 渲染完成后自动刷新文件信息
            render_btn.click(
                get_render_files_info,
                outputs=[srt_info_box, wav_info_box],
            )

    # ── 页面加载时自动恢复数据 ─────────────────────────────────────────────────
    def auto_load_on_start():
        """优先读 translation_v2.json（含翻译），否则读 translation.json。"""
        v2   = os.path.join(WORKSPACE, "translation_v2.json")
        base = os.path.join(WORKSPACE, "translation.json")
        path = v2 if os.path.exists(v2) else base
        if os.path.exists(path):
            status, rows = load_transcript(path)
            srt_info, wav_info = get_render_files_info()
            return status, rows, rows, srt_info, wav_info
        return "ℹ️ 未找到字幕文件，请手动加载", [], [], "", ""

    app.load(
        auto_load_on_start,
        outputs=[transcript_status, seg_df, trans_df, srt_info_box, wav_info_box],
    )


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True)
