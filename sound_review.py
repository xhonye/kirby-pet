"""
猫咪音效人工审查工具
用法：python sound_review.py
然后浏览器打开 http://localhost:8765
"""
import http.server
import json
import os
import sys
import webbrowser
from pathlib import Path
from urllib.parse import unquote

PORT = 8765
SOUNDS_DIR = Path(__file__).parent / "sounds" / "cat"

CATEGORIES = [
    {"id": "random",    "label": "🎲 随机叫",      "desc": "背景随机猫叫，短促自然"},
    {"id": "greeting",  "label": "🎙️ 打招呼",     "desc": "语音唤醒回应，活泼热情"},
    {"id": "pet",       "label": "🖐️ 抚摸/呼噜",  "desc": "被摸时的满足声"},
    {"id": "scare",     "label": "😱 惊吓/嘶嘶",   "desc": "握拳惊吓，防御性"},
    {"id": "sleep",     "label": "😴 睡觉",        "desc": "入睡时的轻柔声"},
    {"id": "growl",     "label": "😾 低吼/警告",   "desc": "不满/警告"},
    {"id": "unused",    "label": "❌ 不用",         "desc": "这个声音不合适"},
]

HTML = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>🐱 猫咪音效审查</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', sans-serif; background: #1a1a2e; color: #eee; min-height: 100vh; display: flex; justify-content: center; padding: 20px; }
.container { max-width: 600px; width: 100%; }
h1 { text-align: center; margin-bottom: 8px; font-size: 1.5em; }
.subtitle { text-align: center; color: #888; margin-bottom: 24px; font-size: 0.9em; }
.progress { text-align: center; color: #aaa; margin-bottom: 16px; font-size: 0.85em; }
.progress-bar { height: 4px; background: #333; border-radius: 2px; margin-bottom: 20px; }
.progress-fill { height: 100%; background: #4ecca3; border-radius: 2px; transition: width 0.3s; }

.card { background: #16213e; border-radius: 12px; padding: 24px; margin-bottom: 16px; }
.sound-name { font-size: 1.4em; font-weight: bold; color: #4ecca3; text-align: center; margin-bottom: 16px; }
.play-btn { display: block; margin: 0 auto 20px; padding: 12px 32px; background: #4ecca3; color: #1a1a2e; border: none; border-radius: 8px; font-size: 1.1em; font-weight: bold; cursor: pointer; transition: all 0.2s; }
.play-btn:hover { background: #3dbb94; transform: scale(1.05); }
.play-btn:active { transform: scale(0.95); }
.play-btn.playing { background: #e84545; }

.categories { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 20px; }
.cat-item { display: flex; align-items: center; gap: 8px; padding: 10px 12px; background: #0f3460; border-radius: 8px; cursor: pointer; transition: background 0.2s; user-select: none; }
.cat-item:hover { background: #1a4a7a; }
.cat-item.selected { background: #1a6a4a; border: 1px solid #4ecca3; }
.cat-item input { display: none; }
.cat-check { width: 20px; height: 20px; border: 2px solid #555; border-radius: 4px; display: flex; align-items: center; justify-content: center; flex-shrink: 0; transition: all 0.2s; }
.cat-item.selected .cat-check { background: #4ecca3; border-color: #4ecca3; }
.cat-item.selected .cat-check::after { content: "✓"; color: #1a1a2e; font-weight: bold; font-size: 14px; }
.cat-label { font-size: 0.95em; }
.cat-desc { font-size: 0.75em; color: #888; }

.nav { display: flex; gap: 12px; }
.nav button { flex: 1; padding: 12px; border: none; border-radius: 8px; font-size: 1em; cursor: pointer; transition: all 0.2s; }
.btn-prev { background: #333; color: #eee; }
.btn-prev:hover { background: #444; }
.btn-next { background: #4ecca3; color: #1a1a2e; font-weight: bold; }
.btn-next:hover { background: #3dbb94; }
.btn-next:disabled { background: #333; color: #666; cursor: not-allowed; }

.result { display: none; }
.result textarea { width: 100%; height: 400px; background: #0f3460; color: #4ecca3; border: 1px solid #333; border-radius: 8px; padding: 16px; font-family: 'Consolas', monospace; font-size: 0.9em; resize: vertical; }
.copy-btn { display: block; margin: 12px auto 0; padding: 10px 24px; background: #4ecca3; color: #1a1a2e; border: none; border-radius: 8px; cursor: pointer; font-weight: bold; }
.copy-btn:hover { background: #3dbb94; }
.restart-btn { display: block; margin: 8px auto 0; padding: 8px 20px; background: transparent; color: #888; border: 1px solid #444; border-radius: 8px; cursor: pointer; font-size: 0.85em; }
.restart-btn:hover { color: #eee; border-color: #888; }
</style>
</head>
<body>
<div class="container">
    <h1>🐱 猫咪音效审查</h1>
    <p class="subtitle">听每个声音，勾选适合的场景，多选OK</p>

    <div id="review-area">
        <div class="progress">
            <span id="progress-text">1 / 9</span>
        </div>
        <div class="progress-bar"><div class="progress-fill" id="progress-fill"></div></div>

        <div class="card">
            <div class="sound-name" id="sound-name">growl_01.mp3</div>
            <button class="play-btn" id="play-btn" onclick="togglePlay()">▶ 播放</button>

            <div class="categories" id="categories"></div>

            <div class="nav">
                <button class="btn-prev" onclick="prevSound()">⬅ 上一个</button>
                <button class="btn-next" onclick="nextSound()">下一个 ➡</button>
            </div>
        </div>
    </div>

    <div class="result" id="result-area">
        <div class="card">
            <h2 style="text-align:center;margin-bottom:16px;">📋 审查结果</h2>
            <textarea id="result-text" readonly></textarea>
            <button class="copy-btn" onclick="copyResult()">📋 复制到剪贴板</button>
            <button class="restart-btn" onclick="restart()">🔄 重新审查</button>
        </div>
    </div>
</div>

<script>
const SOUNDS = __SOUNDS__;
const CATEGORIES = __CATEGORIES__;

let current = 0;
let assignments = {};  // {filename: [cat_id, ...]}
let audio = null;
let isPlaying = false;

// Init
SOUNDS.forEach(f => { assignments[f] = []; });
renderCategories();
updateUI();

function renderCategories() {
    const el = document.getElementById('categories');
    el.innerHTML = CATEGORIES.map(c => `
        <label class="cat-item" id="cat-${c.id}" onclick="toggleCat('${c.id}', event)">
            <input type="checkbox" name="cat" value="${c.id}">
            <div class="cat-check"></div>
            <div>
                <div class="cat-label">${c.label}</div>
                <div class="cat-desc">${c.desc}</div>
            </div>
        </label>
    `).join('');
}

function updateUI() {
    const fname = SOUNDS[current];
    document.getElementById('sound-name').textContent = fname;
    document.getElementById('progress-text').textContent = `${current + 1} / ${SOUNDS.length}`;
    document.getElementById('progress-fill').style.width = `${((current + 1) / SOUNDS.length) * 100}%`;

    // Update category selections
    CATEGORIES.forEach(c => {
        const el = document.getElementById('cat-' + c.id);
        if (assignments[fname].includes(c.id)) {
            el.classList.add('selected');
        } else {
            el.classList.remove('selected');
        }
    });

    // Stop any playing audio
    stopAudio();

    // Update next button text
    const nextBtn = document.querySelector('.btn-next');
    nextBtn.textContent = current >= SOUNDS.length - 1 ? '✅ 完成' : '下一个 ➡';

    // Prev button
    document.querySelector('.btn-prev').style.visibility = current > 0 ? 'visible' : 'hidden';
}

function toggleCat(catId, e) {
    e.preventDefault();
    const fname = SOUNDS[current];
    const idx = assignments[fname].indexOf(catId);
    if (idx >= 0) {
        assignments[fname].splice(idx, 1);
    } else {
        assignments[fname].push(catId);
    }
    updateUI();
}

function togglePlay() {
    if (isPlaying) {
        stopAudio();
    } else {
        playCurrent();
    }
}

function playCurrent() {
    stopAudio();
    const fname = SOUNDS[current];
    audio = new Audio(`/sounds/${fname}`);
    audio.play();
    isPlaying = true;
    const btn = document.getElementById('play-btn');
    btn.textContent = '⏹ 停止';
    btn.classList.add('playing');
    audio.onended = () => {
        isPlaying = false;
        btn.textContent = '▶ 播放';
        btn.classList.remove('playing');
    };
}

function stopAudio() {
    if (audio) {
        audio.pause();
        audio.currentTime = 0;
        audio = null;
    }
    isPlaying = false;
    const btn = document.getElementById('play-btn');
    btn.textContent = '▶ 播放';
    btn.classList.remove('playing');
}

function nextSound() {
    if (current >= SOUNDS.length - 1) {
        showResult();
        return;
    }
    current++;
    updateUI();
    playCurrent();
}

function prevSound() {
    if (current > 0) {
        current--;
        updateUI();
        playCurrent();
    }
}

function showResult() {
    stopAudio();
    document.getElementById('review-area').style.display = 'none';
    document.getElementById('result-area').style.display = 'block';

    // Build result text
    const catMap = {};
    CATEGORIES.forEach(c => { catMap[c.id] = []; });

    SOUNDS.forEach(fname => {
        assignments[fname].forEach(catId => {
            catMap[catId].push(fname);
        });
    });

    let text = '# 猫咪音效审查结果\\n';
    text += `# 审查时间: ${new Date().toLocaleString('zh-CN')}\\n\\n`;

    CATEGORIES.forEach(c => {
        const files = catMap[c.id];
        text += `## ${c.label}\\n`;
        if (files.length === 0) {
            text += '  (无)\\n';
        } else {
            files.forEach(f => { text += `  - ${f}\\n`; });
        }
        text += '\\n';
    });

    // Also show per-file summary
    text += '## 逐文件摘要\\n';
    SOUNDS.forEach(fname => {
        const cats = assignments[fname].map(id => {
            const c = CATEGORIES.find(x => x.id === id);
            return c ? c.label : id;
        });
        text += `  ${fname}: ${cats.length > 0 ? cats.join(', ') : '未分配'}\\n`;
    });

    document.getElementById('result-text').value = text.replace(/\\\\n/g, '\\n');
}

function copyResult() {
    const textarea = document.getElementById('result-text');
    textarea.select();
    document.execCommand('copy');
    const btn = document.querySelector('.copy-btn');
    btn.textContent = '✅ 已复制';
    setTimeout(() => { btn.textContent = '📋 复制到剪贴板'; }, 2000);
}

function restart() {
    current = 0;
    SOUNDS.forEach(f => { assignments[f] = []; });
    document.getElementById('review-area').style.display = 'block';
    document.getElementById('result-area').style.display = 'none';
    updateUI();
}

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    if (document.getElementById('result-area').style.display === 'block') return;
    if (e.key === ' ' || e.key === 'Enter') { e.preventDefault(); togglePlay(); }
    if (e.key === 'ArrowRight' || e.key === 'n') nextSound();
    if (e.key === 'ArrowLeft' || e.key === 'p') prevSound();
    if (e.key >= '1' && e.key <= '7') {
        const idx = parseInt(e.key) - 1;
        if (idx < CATEGORIES.length) {
            const fname = SOUNDS[current];
            const catId = CATEGORIES[idx].id;
            const i = assignments[fname].indexOf(catId);
            if (i >= 0) assignments[fname].splice(i, 1);
            else assignments[fname].push(catId);
            updateUI();
        }
    }
});
</script>
</body>
</html>"""


def load_reviewed():
    """Load already-reviewed filenames from CSV"""
    csv_path = Path(__file__).parent / "sounds" / "sound_review.csv"
    reviewed = set()
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            next(f)  # skip header
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    reviewed.add(parts[0])
    return reviewed

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            reviewed = load_reviewed()
            sounds = sorted([
                f for f in os.listdir(SOUNDS_DIR)
                if f.endswith(".mp3") and os.path.isfile(SOUNDS_DIR / f) and f not in reviewed
            ])
            html = HTML.replace("__SOUNDS__", json.dumps(sounds))
            html = html.replace("__CATEGORIES__", json.dumps(CATEGORIES))
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html.encode("utf-8"))
        elif self.path.startswith("/sounds/"):
            fname = unquote(self.path[8:])
            fpath = SOUNDS_DIR / fname
            if fpath.exists() and fpath.is_file():
                self.send_response(200)
                self.send_header("Content-Type", "audio/mpeg")
                self.send_header("Content-Length", str(fpath.stat().st_size))
                self.end_headers()
                with open(fpath, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404, f"File not found: {fname}")
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass  # Suppress logs


if __name__ == "__main__":
    os.chdir(Path(__file__).parent)
    server = http.server.HTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://localhost:{PORT}"
    print(f"🐱 猫咪音效审查工具")
    print(f"   打开浏览器: {url}")
    print(f"   按 Ctrl+C 退出")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已退出")
        server.server_close()
