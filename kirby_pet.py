"""
Kirby Desktop Pet — v3
- 卡比风格大眼睛（椭圆+蓝色虹膜+瞳孔）
- 倒三角嘴巴（张嘴时变大）
- 手势识别：捏合→展开 = pet
- 语音唤醒：喊"卡比"回应
- 随机卡比声音
"""

import pygame
import cv2
import numpy as np
import threading
import time
import random
import math
import sys
import os

# ── 配置 ──────────────────────────────────────────────
WIDTH, HEIGHT = 500, 500
BG_COLOR = (30, 30, 40)
FPS = 60

# 卡比眼睛参数（更像真卡比）
EYE_COLOR = (255, 255, 255)      # 眼白
IRIS_COLOR = (50, 100, 200)      # 蓝色虹膜（卡比经典色）
PUPIL_COLOR = (20, 20, 60)       # 深色瞳孔
HIGHLIGHT_COLOR = (255, 255, 255) # 高光
BLUSH_COLOR = (255, 130, 150, 180) # 腮红

EYE_W = 80                       # 眼睛宽度
EYE_H = 110                      # 眼睛高度
EYE_SPACING = 70                 # 两眼间距
EYE_Y = HEIGHT // 2 - 30         # 眼睛Y位置
EYE_LEFT_X = WIDTH // 2 - EYE_SPACING
EYE_RIGHT_X = WIDTH // 2 + EYE_SPACING
PUPIL_TRACK_RANGE = 25           # 瞳孔跟踪范围

# 倒三角嘴巴参数
MOUTH_Y = EYE_Y + 85
MOUTH_W = 30                     # 嘴巴半宽
MOUTH_H_CLOSED = 8               # 闭嘴时高度
MOUTH_H_OPEN = 40                # 张嘴时高度
MOUTH_COLOR = (180, 60, 80)      # 嘴巴颜色（深红）
TONGUE_COLOR = (220, 100, 100)   # 舌头颜色

# 眨眼
BLINK_INTERVAL = (2, 6)
BLINK_DURATION = 0.15

# 打瞌睡
SLEEP_TIMEOUT = 15
SNOOZE_EYE_H = 20

# 声音
SAMPLE_RATE = 22050
RANDOM_SOUND_INTERVAL = (6, 18)

# 手势检测（手掌摇晃 = pet）
PALM_HISTORY_LEN = 15            # 记录最近 N 帧手掌位置
SHAKE_THRESHOLD = 0.08           # 摇晃距离阈值（归一化坐标）
SHAKE_MIN_COUNT = 3              # 最少方向变换次数才算摇晃
GESTURE_COOLDOWN = 2.0           # 两次 pet 间隔

# ── 全局状态 ────────────────────────────────────────────
face_x = None
face_y = None
last_face_time = 0
state = "idle"
pupil_offset = [0.0, 0.0]
target_pupil = [0.0, 0.0]
wander_timer = 0
blink_timer = 0
is_blinking = False
blink_end = 0
next_blink = random.uniform(*BLINK_INTERVAL)

# 嘴巴状态
mouth_open = 0.0
mouth_target = 0.0
mouth_event_until = 0

# 手势状态
palm_history = []                # 手掌位置历史 [(x,y,timestamp), ...]
pet_cooldown_until = 0           # pet 冷却
eye_pet_close_until = 0          # pet 时闭眼

# 语音唤醒状态
voice_wake_until = 0
voice_event_type = None           # "hi" or None

# 下一次随机声音
next_random_sound = random.uniform(*RANDOM_SOUND_INTERVAL)

# 摄像头
camera_status = "init"
camera_debug_frame = None
detection_count = 0

# ── 音效加载 ────────────────────────────────────────────
def load_sounds():
    """加载猫咪音效"""
    import glob
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cat_dir = os.path.join(script_dir, "sounds", "cat")
    loaded = {}
    
    def load_one(path):
        try:
            return pygame.mixer.Sound(path)
        except Exception as e:
            print(f"  ❌ load failed: {path}: {e}")
            return None
    
    # 猫咪音效映射
    sound_map = {
        "purr": ["purr_01.mp3", "purr_02.mp3", "purr_03.mp3"],      # 呼噜（被摸/满足）
        "meow": ["meow_01.mp3", "meow_02.mp3", "meow_03.mp3", "meow_04.mp3"],  # 喵叫（打招呼/回应）
        "hiss": ["hiss_01.mp3", "hiss_02.mp3"],      # 嘶嘶（惊吓/不爽）
        "growl": ["growl_01.mp3"],                    # 低吼（生气/警告）
    }
    
    for name, files in sound_map.items():
        sounds = []
        for fname in files:
            path = os.path.join(cat_dir, fname)
            if os.path.exists(path):
                s = load_one(path)
                if s:
                    sounds.append(s)
        if sounds:
            loaded[name] = sounds
            print(f"  ✅ {name}: {len(sounds)} sounds")
        else:
            print(f"  ⚠️ {name}: no sounds found")
    
    return loaded

# ── 语音识别线程 ────────────────────────────────────────
def voice_recognition_loop():
    """语音识别：监听麦克风，检测到喊"卡比"时触发回应
    使用 Google Web Speech API（免费，需联网）"""
    global voice_wake_until, mouth_event_until
    try:
        import speech_recognition as sr
    except ImportError:
        print("[WARN] speech_recognition 未安装，语音功能禁用")
        print("[WARN] 运行: pip install SpeechRecognition pyaudio")
        return
    
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300       # 音量阈值
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.8        # 停顿判定
    
    try:
        mic = sr.Microphone()
    except Exception as e:
        print(f"[WARN] 麦克风不可用: {e}")
        return
    
    print("[INFO] 语音识别启动（Google API）")
    print("[INFO] 喊卡比试试！")
    
    # 校准环境噪音
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
    
    while True:
        try:
            with mic as source:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=3)
            
            text = recognizer.recognize_google(audio, language="zh-CN")
            print(f"[VOICE] {text}")
            
            # 模糊匹配打招呼
            hi_keywords = ["hi", "嗨", "嘿", "hello", "你好", "哈喽", "hey", "猫咪", "猫猫", "喵"]
            if any(kw in text.lower() for kw in hi_keywords):
                print("[VOICE] ★ 打招呼！")
                voice_wake_until = time.time() + 2.0
                mouth_event_until = time.time() + 1.5
                # 标记为 hi 事件，播放 hi 音效
                voice_event_type = "hi"
            
            # 模糊匹配叫猫咪
            kirby_keywords = ["猫咪", "猫猫", "小猫", "喵", "cat", "kitty"]
            if any(kw in text.lower() for kw in kirby_keywords):
                print("[VOICE] ★ 叫猫咪！")
                voice_wake_until = time.time() + 2.0
                mouth_event_until = time.time() + 1.5
                voice_event_type = "hi"  # 卡比也用 hi 音效回应
                
        except sr.WaitTimeoutError:
            pass
        except sr.UnknownValueError:
            pass
        except sr.RequestError as e:
            print(f"[WARN] Google 语音服务错误: {e}")
            print("[WARN] 检查网络连接，或等待一会儿重试")
            time.sleep(5)
        except OSError as e:
            print(f"[ERROR] 麦克风错误: {e}")
            time.sleep(3)
        except Exception as e:
            print(f"[WARN] 语音异常: {type(e).__name__}: {e}")
            time.sleep(1)

# ── 摄像头检测线程（人脸+手势）──────────────────────────
def camera_detection_loop():
    global face_x, face_y, last_face_time, camera_status
    global camera_debug_frame, detection_count
    global palm_history, pet_cooldown_until, eye_pet_close_until
    
    import mediapipe as mp_lib
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    face_model = os.path.join(script_dir, "models", "blaze_face_short_range.tflite")
    hand_model = os.path.join(script_dir, "models", "hand_landmarker.task")
    
    # 人脸检测器
    if not os.path.exists(face_model):
        camera_status = "failed"
        print(f"[ERROR] 人脸模型不存在: {face_model}")
        return
    
    face_opts = vision.FaceDetectorOptions(
        base_options=mp_python.BaseOptions(model_asset_path=face_model),
        min_detection_confidence=0.4,
        running_mode=vision.RunningMode.VIDEO,
    )
    face_detector = vision.FaceDetector.create_from_options(face_opts)
    
    # 手部检测器
    hand_detector = None
    if os.path.exists(hand_model):
        hand_opts = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=hand_model),
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=vision.RunningMode.VIDEO,
        )
        hand_detector = vision.HandLandmarker.create_from_options(hand_opts)
        print("[INFO] 手部检测已加载")
    
    print("[INFO] 正在打开摄像头...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        camera_status = "failed"
        print("[ERROR] 摄像头未打开！")
        return
    
    camera_status = "opened"
    print("[INFO] 摄像头已打开 | 按 Q 关闭调试窗口")
    
    
    while True:
        ret, frame = cap.read()
        if not ret:
            camera_status = "failed"
            time.sleep(0.1)
            continue
        
        small = cv2.resize(frame, (320, 240))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mp_image = mp_lib.Image(image_format=mp_lib.ImageFormat.SRGB, data=rgb)
        ts = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        debug = small.copy()
        now = time.time()
        
        # 人脸检测
        face_result = face_detector.detect_for_video(mp_image, ts)
        if face_result.detections:
            det = face_result.detections[0]
            bbox = det.bounding_box
            cx = (bbox.origin_x + bbox.width / 2) / 320.0
            cy = (bbox.origin_y + bbox.height / 2) / 240.0
            face_x, face_y = cx, cy
            last_face_time = now
            detection_count += 1
            camera_status = "tracking"
            
            x1, y1 = bbox.origin_x, bbox.origin_y
            x2, y2 = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
            cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(debug, "FACE", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            # 手势检测：手掌摇晃 = pet
            if hand_detector is not None:
                hand_result = hand_detector.detect_for_video(mp_image, ts)
                if hand_result.hand_landmarks:
                    hand = hand_result.hand_landmarks[0]
                    
                    # 只画关键点（21个点，无线条）
                    for lm in hand:
                        px, py = int(lm.x * 320), int(lm.y * 240)
                        cv2.circle(debug, (px, py), 3, (255, 200, 0), -1)
                    
                    # 手掌中心（手腕+中指根部中点）
                    wrist = hand[0]
                    mid_base = hand[9]
                    palm_cx = (wrist.x + mid_base.x) / 2
                    palm_cy = (wrist.y + mid_base.y) / 2
                    px, py = int(palm_cx * 320), int(palm_cy * 240)
                    cv2.circle(debug, (px, py), 6, (0, 255, 255), -1)
                    cv2.putText(debug, "PALM", (px + 10, py), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                    
                    # 记录手掌位置
                    palm_history.append((palm_cx, palm_cy, now))
                    if len(palm_history) > PALM_HISTORY_LEN:
                        palm_history.pop(0)
                    
                    # 检测摇晃
                    if len(palm_history) >= 5:
                        dx_list = [palm_history[j][0] - palm_history[j-1][0] for j in range(1, len(palm_history))]
                        dy_list = [palm_history[j][1] - palm_history[j-1][1] for j in range(1, len(palm_history))]
                        x_changes = sum(1 for k in range(1, len(dx_list)) if dx_list[k] * dx_list[k-1] < 0)
                        y_changes = sum(1 for k in range(1, len(dy_list)) if dy_list[k] * dy_list[k-1] < 0)
                        total_changes = x_changes + y_changes
                        total_dist = sum(math.sqrt(dx**2 + dy**2) for dx, dy in zip(dx_list, dy_list))
                        
                        shake_label = "PET!" if total_changes >= SHAKE_MIN_COUNT and total_dist > SHAKE_THRESHOLD else f"shake:{total_changes} d:{total_dist:.2f}"
                        color = (0, 255, 255) if "PET" in shake_label else (200, 200, 200)
                        cv2.putText(debug, shake_label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                        
                        if total_changes >= SHAKE_MIN_COUNT and total_dist > SHAKE_THRESHOLD:
                            if now > pet_cooldown_until:
                                print(f"[GESTURE] Pet! (shake {total_changes}x)")
                                eye_pet_close_until = now + 1.5
                                pet_cooldown_until = now + GESTURE_COOLDOWN
                                palm_history.clear()
                                cv2.putText(debug, "PET!", (120, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                else:
                    palm_history.clear()
        else:
            face_x, face_y = None, None
            camera_status = "no_face"
            cv2.putText(debug, "NO FACE", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        camera_debug_frame = debug
        cv2.imshow("Kirby Debug - Camera", debug)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(0.03)

# ── 绘制卡比风格眼睛 ────────────────────────────────────
# 呼吸动画
_breath_t = 0.0

def draw_eyes(screen, eye_open_ratio=1.0):
    global _breath_t
    _breath_t += 0.03
    breath_scale = 1.0 + 0.012 * math.sin(_breath_t * 1.5)  # 微小呼吸缩放
    
    for ex in [EYE_LEFT_X, EYE_RIGHT_X]:
        if eye_open_ratio < 0.2:
            # 月牙眯眼笑（被摸时）
            draw_crescent_eye(screen, ex, EYE_Y, int(EYE_W * 0.7))
        else:
            # 正常眼睛
            w = int(EYE_W * breath_scale)
            h = max(5, int(EYE_H * eye_open_ratio * breath_scale))
            eye_rect = pygame.Rect(0, 0, w, h)
            eye_rect.center = (ex, EYE_Y)
            pygame.draw.ellipse(screen, EYE_COLOR, eye_rect)
            pygame.draw.ellipse(screen, (200, 200, 220), eye_rect, 2)
            
            if eye_open_ratio > 0.25:
                # 蓝色虹膜
                iris_w = int(w * 0.65)
                iris_h = int(h * 0.7)
                iris_rect = pygame.Rect(0, 0, iris_w, iris_h)
                ix = ex + int(pupil_offset[0] * PUPIL_TRACK_RANGE)
                iy = EYE_Y + int(pupil_offset[1] * PUPIL_TRACK_RANGE * 0.5)
                iris_rect.center = (ix, iy)
                pygame.draw.ellipse(screen, IRIS_COLOR, iris_rect)
                
                # 深色瞳孔
                pr = max(3, int(iris_w * 0.35 * eye_open_ratio))
                pygame.draw.circle(screen, PUPIL_COLOR, (ix, iy), pr)
                
                # 多层高光
                # 大高光（左上）
                hx1 = ix - int(pr * 0.5)
                hy1 = iy - int(pr * 0.5)
                hr1 = max(3, int(pr * 0.45))
                pygame.draw.circle(screen, HIGHLIGHT_COLOR, (hx1, hy1), hr1)
                # 小高光（右下）
                hx2 = ix + int(pr * 0.3)
                hy2 = iy + int(pr * 0.35)
                hr2 = max(2, int(pr * 0.2))
                pygame.draw.circle(screen, (255, 255, 255, 180), (hx2, hy2), hr2)
    
    # 腮红（呼吸动画）
    blush_y = EYE_Y + 45
    blush_w = int(50 * breath_scale)
    blush_h = int(25 * breath_scale)
    blush_surf = pygame.Surface((blush_w, blush_h), pygame.SRCALPHA)
    pygame.draw.ellipse(blush_surf, BLUSH_COLOR, (0, 0, blush_w, blush_h))
    screen.blit(blush_surf, (EYE_LEFT_X - 75, blush_y))
    screen.blit(blush_surf, (EYE_RIGHT_X + 25, blush_y))

def draw_crescent_eye(screen, cx, cy, w):
    """月牙形眯眼笑"""
    # 画一条向上弯的弧线
    points = []
    for i in range(20):
        t = i / 19.0
        x = cx - w + t * w * 2
        y = cy + math.sin(t * math.pi) * 12 - 5
        points.append((x, y))
    if len(points) >= 2:
        pygame.draw.lines(screen, (60, 60, 100), False, points, 4)

# ── 绘制嘴巴（圆角倒三角+舌头）──────────────────────────
_mouth_breathe = 0.0

def draw_mouth(screen, open_ratio):
    global _mouth_breathe
    _mouth_breathe += 0.02
    cx = WIDTH // 2
    
    if open_ratio < 0.15:
        # 闭嘴：小弧线微笑
        points = []
        for i in range(20):
            t = i / 19.0
            x = cx - MOUTH_W + t * MOUTH_W * 2
            y = MOUTH_Y + math.sin(t * math.pi) * 6
            points.append((x, y))
        if len(points) >= 2:
            pygame.draw.lines(screen, MOUTH_COLOR, False, points, 3)
    else:
        # 张嘴：圆角倒三角
        w = int(MOUTH_W * (1.0 + 0.5 * open_ratio))
        h = int(MOUTH_H_CLOSED + (MOUTH_H_OPEN - MOUTH_H_CLOSED) * open_ratio)
        
        # 深色喉咙（底层）
        throat_pts = [
            (cx - w + 4, MOUTH_Y),
            (cx + w - 4, MOUTH_Y),
            (cx, MOUTH_Y + h - 2),
        ]
        pygame.draw.polygon(screen, (80, 30, 40), throat_pts)
        
        # 粉色嘴巴主体
        mouth_pts = [
            (cx - w, MOUTH_Y - 2),
            (cx + w, MOUTH_Y - 2),
            (cx, MOUTH_Y + h),
        ]
        pygame.draw.polygon(screen, MOUTH_COLOR, mouth_pts)
        
        # 舌头（张得够大时）
        if open_ratio > 0.4:
            tw = int(w * 0.45)
            th = int(h * 0.35)
            ty = MOUTH_Y + int(h * 0.45)
            # 圆润舌头
            tongue_rect = pygame.Rect(cx - tw, ty, tw * 2, th)
            pygame.draw.ellipse(screen, TONGUE_COLOR, tongue_rect)

# ── 爱心粒子系统 ─────────────────────────────────────────
_hearts = []  # [(x, y, vx, vy, life, max_life), ...]

def spawn_hearts(count=5):
    """在卡比头顶生成爱心粒子"""
    import random as _r
    for _ in range(count):
        x = WIDTH // 2 + _r.randint(-60, 60)
        y = EYE_Y - 80 + _r.randint(-20, 10)
        vx = _r.uniform(-0.8, 0.8)
        vy = _r.uniform(-2.5, -1.0)
        life = _r.uniform(1.0, 2.0)
        _hearts.append([x, y, vx, vy, life, life])

def update_hearts(dt):
    """更新并绘制爱心"""
    to_remove = []
    for h in _hearts:
        h[0] += h[2]  # x += vx
        h[1] += h[3]  # y += vy
        h[3] += 0.5 * dt  # gravity
        h[4] -= dt       # life -= dt
        
        if h[4] <= 0:
            to_remove.append(h)
            continue
        
        alpha = int(255 * (h[4] / h[5]))
        size = int(8 * (h[4] / h[5]))
        if size < 2:
            continue
        
        # 画小爱心
        sx, sy = int(h[0]), int(h[1])
        heart_surf = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
        color = (255, 100, 120, alpha)
        # 简单爱心：两个圆+一个三角
        pygame.draw.circle(heart_surf, color, (size // 2, size // 2), size // 2)
        pygame.draw.circle(heart_surf, color, (size + size // 2, size // 2), size // 2)
        pygame.draw.polygon(heart_surf, color, [(0, size), (size, size * 2 - 1), (size * 2, size)])
        screen_rect = pygame.Rect(sx - size, sy - size, size * 2, size * 2)
        # 需要 screen 引用，改用全局
        pass
    
    for h in to_remove:
        _hearts.remove(h)

def draw_hearts(screen, dt):
    """更新并绘制爱心粒子"""
    global _hearts
    to_remove = []
    for h in _hearts:
        h[0] += h[2]
        h[1] += h[3]
        h[3] += 30 * dt  # gravity
        h[4] -= dt
        
        if h[4] <= 0:
            to_remove.append(h)
            continue
        
        alpha = max(0, min(255, int(255 * (h[4] / h[5]))))
        size = max(2, int(8 * (h[4] / h[5])))
        sx, sy = int(h[0]), int(h[1])
        
        heart_surf = pygame.Surface((size * 3, size * 3), pygame.SRCALPHA)
        color = (255, 100, 130, alpha)
        # 两个圆 + 三角 = 爱心
        r = size // 2
        if r < 1: r = 1
        pygame.draw.circle(heart_surf, color, (r + 1, r + 1), r)
        pygame.draw.circle(heart_surf, color, (r * 2 + 1, r + 1), r)
        pygame.draw.polygon(heart_surf, color, [(0, r + 1), (r * 1.5 + 1, r * 3), (r * 3 + 1, r + 1)])
        screen.blit(heart_surf, (sx - size, sy - size))
    
    for h in to_remove:
        _hearts.remove(h)

# ── 头顶小脚丫 ─────────────────────────────────────────
def draw_feet(screen):
    """卡比头顶标志性小脚丫"""
    foot_y = EYE_Y - EYE_H - 15
    foot_color = (220, 60, 80)  # 红色
    
    for fx in [WIDTH // 2 - 20, WIDTH // 2 + 12]:
        # 小椭圆脚掌
        foot_rect = pygame.Rect(0, 0, 18, 10)
        foot_rect.center = (fx, foot_y)
        pygame.draw.ellipse(screen, foot_color, foot_rect)

# ── 状态绘制 ────────────────────────────────────────────
def draw_state(screen, font, state_text, fps_val):
    text = font.render(f"{state_text}  FPS:{fps_val:.0f}", True, (100, 100, 120))
    screen.blit(text, (10, 10))
    
    cam_colors = {"init": (180,180,100), "opened": (100,180,100), "failed": (255,80,80), "no_face": (180,130,80), "tracking": (80,220,80)}
    cam_labels = {
        "init": "CAM: init...", "opened": "CAM: waiting...", "failed": "CAM: FAILED",
        "no_face": "CAM: no face", "tracking": f"FACE #{detection_count}",
    }
    cam_text = font.render(cam_labels.get(camera_status, f"CAM: {camera_status}"), True, cam_colors.get(camera_status, (100,100,100)))
    screen.blit(cam_text, (10, HEIGHT - 25))

# ── 主循环 ──────────────────────────────────────────────
def main():
    global state, pupil_offset, target_pupil
    global blink_timer, is_blinking, blink_end, next_blink
    global wander_timer, mouth_open, mouth_target
    global mouth_event_until, eye_pet_close_until
    global voice_wake_until, voice_event_type, next_random_sound
    
    pygame.init()
    sound_enabled = False
    try:
        pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2)
        sound_enabled = True
    except pygame.error as e:
        print(f"[WARN] 音频失败: {e}")
    
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Cat Pet")
    try:
        import ctypes
        hwnd = pygame.display.get_wm_info()["window"]
        ctypes.windll.user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, 0x0001 | 0x0002)
    except Exception:
        pass
    
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 14)
    
    sounds = {}
    if sound_enabled:
        print("[INFO] 加载音效...")
        sounds = load_sounds()
    
    threading.Thread(target=camera_detection_loop, daemon=True).start()
    threading.Thread(target=voice_recognition_loop, daemon=True).start()
    
    eye_open = 1.0
    sleep_level = 0
    running = True
    
    while running:
        dt = clock.tick(FPS) / 1000.0
        now = time.time()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # 状态机
        if face_x is not None:
            state = "tracking"
            sleep_level = 0
            target_pupil[0] = (0.5 - face_x) * 2
            if face_y is not None:
                target_pupil[1] = (face_y - 0.5) * 1.5
            wander_timer = now
        else:
            if now - wander_timer > SLEEP_TIMEOUT:
                state = "sleeping"
                sleep_level = min(1.0, sleep_level + dt * 0.3)
            else:
                state = "idle"
                if random.random() < 0.02:
                    target_pupil[0] = random.uniform(-0.8, 0.8)
                    target_pupil[1] = random.uniform(-0.3, 0.3)
        
        for i in range(2):
            pupil_offset[i] += (target_pupil[i] - pupil_offset[i]) * min(1, dt * 8)
        
        # 眨眼
        blink_timer += dt
        if is_blinking:
            if now > blink_end:
                is_blinking = False
                next_blink = random.uniform(*BLINK_INTERVAL)
                blink_timer = 0
        else:
            if blink_timer > next_blink:
                is_blinking = True
                blink_end = now + BLINK_DURATION
        
        # 眼睛睁开程度
        if now < eye_pet_close_until:
            eye_open += (0.15 - eye_open) * min(1, dt * 10)  # pet 时闭眼
        elif is_blinking:
            eye_open = max(0.05, eye_open - dt * 15)
        elif state == "sleeping":
            eye_open += ((SNOOZE_EYE_H / EYE_H) - eye_open) * dt * 2
            # 睡觉时播放打哈欠音效
            if not hasattr(main, '_sleep_sound_played') or not main._sleep_sound_played:
                if sound_enabled and sounds.get("purr"):
                    s = random.choice(sounds["purr"])
                    s.set_volume(0.3)
                    s.play()
                    main._sleep_sound_played = True
        else:
            eye_open += (1.0 - eye_open) * dt * 10
            if hasattr(main, '_sleep_sound_played'):
                main._sleep_sound_played = False
        
        # 嘴巴
        mouth_target = 0.0
        if now < mouth_event_until:
            mouth_target = 0.8
        elif now < voice_wake_until:
            mouth_target = 0.5
        mouth_open += (mouth_target - mouth_open) * min(1, dt * 12)
        
        # pet 时发声 + 冒爱心
        if now < eye_pet_close_until and now > eye_pet_close_until - 1.4:
            if sound_enabled and sounds.get("purr"):
                s = random.choice(sounds["purr"])
                s.set_volume(0.5)
                s.play()
            spawn_hearts(6)
            eye_pet_close_until = now - 1
        
        # 语音唤醒发声
        if now < voice_wake_until and now > voice_wake_until - 1.9 and sound_enabled:
            if voice_event_type == "hi" and sounds.get("meow"):
                s = random.choice(sounds["meow"])
                s.set_volume(0.6)
                s.play()
            elif sounds.get("meow"):
                s = random.choice(sounds["meow"])
                s.set_volume(0.5)
                s.play()
            voice_wake_until = now - 1
            voice_event_type = None
        
        # 随机声音
        next_random_sound -= dt
        if next_random_sound <= 0 and sound_enabled:
            pool = random.choice(["meow", "meow", "purr", "purr"])
            if sounds.get(pool):
                s = random.choice(sounds[pool])
                s.set_volume(random.uniform(0.2, 0.4))
                s.play()
            next_random_sound = random.uniform(*RANDOM_SOUND_INTERVAL)
        
        # 绘制
        screen.fill(BG_COLOR)
        draw_feet(screen)
        draw_eyes(screen, eye_open)
        draw_mouth(screen, mouth_open)
        draw_hearts(screen, dt)
        draw_state(screen, font, state, clock.get_fps())
        pygame.display.flip()
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
