# 卡比音效资源汇总

## 🎮 GBA 游戏解包资源

### sounds-resource.com（最推荐）
- **Kirby & The Amazing Mirror (镜之迷宫)**
  - URL: https://www.sounds-resource.com/game_boy_advance/kirbyandtheamazingmirror/
  - 包含: Voice, SFX, Music 完整解包
  - 下载方式: 浏览器访问 → 选择类别 → 下载 ZIP

- **Kirby: Nightmare in Dream Land**
  - URL: https://www.sounds-resource.com/game_boy_advance/kirbynightmareindreamland/
  - 包含: Voice, SFX

- **Kirby Squeak Squad (DS)**
  - URL: https://www.sounds-resource.com/ds/kirbysqueaksquad/

- **Kirby Super Star (SNES)**
  - URL: https://www.sounds-resource.com/snes/kirbysuperstar/

- **Kirby Air Ride (GC)**
  - URL: https://www.sounds-resource.com/gamecube/kirbyairride/

### khinsider.com（游戏原声带）
- 搜索: https://downloads.khinsider.com/search?search=kirby
- 主要是音乐，部分包含 SFX

## 🌐 在线音效库

### myinstants.com（即时播放+下载）
- 搜索: https://www.myinstants.com/search/?name=kirby
- 已确认可用: poyo.mp3
- 需要浏览器访问下载

### freesound.org（CC 授权音效）
- 搜索: https://freesound.org/search/?q=kirby
- 需要注册账号下载

### mixkit.co（免费音效）
- 搜索: https://mixkit.co/free-sound-effects/?q=kirby
- 无 Kirby 专用，但有类似可爱音效

## 📺 视频提取方案

### YouTube/B站 游戏视频提取
1. 搜索: "Kirby Super Star SFX", "星之卡比 镜之迷宫 音效"
2. 下载视频: yt-dlp + ffmpeg
3. 用 Audacity 截取片段

### 推荐视频关键词
- "Kirby sound effects compilation"
- "星之卡比 全音效"
- "Kirby voice clips all"
- "Kirby Super Star all abilities sounds"

## 🔧 提取工具

### 必装工具
```bash
pip install yt-dlp
winget install ffmpeg
```

### 提取步骤
```bash
# 下载音频
yt-dlp -x --audio-format wav "VIDEO_URL" -o kirby_raw.wav

# 用 Audacity 打开，截取片段，导出为 WAV
# 替换 sounds/ 目录下对应文件
```

## 📁 需要的音效文件

| 文件名 | 用途 | 来源建议 |
|--------|------|----------|
| poyo.wav | 卡比叫声 | 游戏 Voice 类别 |
| inhale.wav | 吸气声 | 游戏 SFX 类别 |
| hurt.wav | 受伤声 | 游戏 Voice/SFX |
| victory.wav | 通关音效 | 游戏 Stage Clear |
| pet.wav | 被摸反应 | 游戏 Sleep/Love |
| happy.wav | 开心声 | 游戏 Menu Select |

## 💡 快速方案

最快获取真实卡比音效的方式：
1. 访问 sounds-resource.com 的 Kirby & The Amazing Mirror 页面
2. 下载 Voice 和 SFX 类别的 ZIP
3. 解压后找到需要的 WAV 文件
4. 重命名后替换 sounds/ 目录下的文件
