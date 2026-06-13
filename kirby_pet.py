"""
Cat Desktop Pet — v4
- 猫咪脸型（圆脸+三角耳+绿瞳竖瞳）
- W 形猫嘴 + 张嘴椭圆
- 胡须 + 腮红 + 鼻子
- 猫爪（底部肉垫）
- 手势识别：张手摇晃 = pet，握拳 = 惊吓
- 语音唤醒：喊"猫咪"回应
- 猫咪音效（purr/meow/hiss）
"""

import pygame
import cv2
import numpy as np
import threading
import time
import random
import datetime
import math
import sys
import os

# ── 配置 ──────────────────────────────────────────────
WIDTH, HEIGHT = 500, 500
BG_COLOR = (30, 30, 40)
FPS = 60

# 猫咪参数
FACE_COLOR = (255, 220, 180)     # 猫咪脸蛋颜色（暖米色）
FACE_SHADOW = (230, 190, 150)     # 脸部阴影色
EAR_COLOR = (255, 200, 160)      # 耳朵外色
EAR_INNER_COLOR = (255, 150, 160) # 耳朵内粉色
EAR_TUFT_COLOR = (255, 230, 200)  # 耳尖绒毛
NOSE_COLOR = (255, 140, 150)     # 鼻子粉色
WHISKER_COLOR = (200, 200, 200)  # 胡须颜色
PAW_COLOR = (255, 210, 170)      # 爪子颜色
PAW_PAD_COLOR = (220, 140, 140)  # 肉垫颜色
TAIL_COLOR = (255, 210, 170)     # 尾巴颜色

EYE_COLOR = (255, 255, 255)      # 眼白
IRIS_COLOR = (80, 180, 80)       # 绿色虹膜（猫经典色）
PUPIL_COLOR = (20, 20, 20)       # 黑色瞳孔（竖瞳）
HIGHLIGHT_COLOR = (255, 255, 255) # 高光
BLUSH_COLOR = (255, 150, 160, 120) # 腮红

EYE_W = 55                       # 猫眼宽度（更扁）
EYE_H = 65                       # 猫眼高度（更扁）
EYE_SPACING = 80                 # 两眼间距
EYE_Y = HEIGHT // 2 - 20         # 眼睛Y位置
EYE_LEFT_X = WIDTH // 2 - EYE_SPACING
EYE_RIGHT_X = WIDTH // 2 + EYE_SPACING
PUPIL_TRACK_RANGE = 20           # 瞳孔跟踪范围

# 鼻子
NOSE_Y = EYE_Y + 55
NOSE_SIZE = 12

# 嘴巴参数
MOUTH_Y = NOSE_Y + 15
MOUTH_W = 25                     # 嘴巴半宽
MOUTH_H_CLOSED = 5               # 闭嘴时高度
MOUTH_H_OPEN = 30                # 张嘴时高度
MOUTH_COLOR = (180, 80, 90)      # 嘴巴颜色
TONGUE_COLOR = (255, 130, 130)   # 舌头颜色

# 耳朵
EAR_Y = EYE_Y - 70
EAR_W = 40
EAR_H = 55

# 眨眼
BLINK_INTERVAL = (2, 6)
BLINK_DURATION = 0.15

# 打瞌睡
SLEEP_TIMEOUT = 15
SNOOZE_EYE_H = 20

# 声音
SAMPLE_RATE = 22050
RANDOM_SOUND_INTERVAL = (60, 120)  # 随机叫声间隔：60-120秒

# 手势检测（手掌摇晃 = pet）
PALM_HISTORY_LEN = 25            # 记录最近 N 帧手掌位置（更长观察窗口）
SHAKE_THRESHOLD = 0.20           # 摇晃距离阈值（归一化坐标，需明显移动）
SHAKE_MIN_COUNT = 6              # 最少方向变换次数才算摇晃
GESTURE_COOLDOWN = 8.0           # pet 手势冷却
FIST_THRESHOLD = 0.10              # 握拳判定阈值（更严格，避免误判）

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
hiss_cooldown_until = 0           # 惊吓音效冷却
# 统一音效播放器 — 每个 category 有独立冷却，避免重复叠加
_last_sound_play = {}              # {category: timestamp}
SOUND_COOLDOWNS = {                # 各类音效最短间隔（秒）
    "purr": 4.0,
    "meow": 3.0,
    "hiss": 2.0,
    "growl": 2.0,
}

def play_cat_sound(category, volume=0.5, trigger="unknown"):
    """统一播放入口：带冷却 + 日志"""
    now = time.time()
    cd = SOUND_COOLDOWNS.get(category, 3.0)
    if now - _last_sound_play.get(category, 0) < cd:
        return False  # 冷却中，跳过
    if not sound_enabled or not sounds.get(category):
        return False
    s, fname = random.choice(sounds[category])
    s.set_volume(volume)
    s.play()
    _last_sound_play[category] = now
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {trigger} → {category}/{fname} (vol={volume:.1f})", flush=True)
    return True

# 语音唤醒状态
voice_wake_until = 0
voice_event_type = None           # "hi" or None
voice_cooldown_until = 0          # 语音音效冷却

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
        "purr": ["purr_01.mp3", "purr_02.mp3"],       # 呼噜（抚摸/满足/随机）
        "meow": ["meow_01.mp3", "meow_03.mp3", "meow_04.mp3"],  # 喵叫（打招呼/随机叫）
        "hiss": ["hiss_01.mp3", "hiss_02.mp3", "hiss_03.mp3", "hiss_04.mp3", "meow_02.mp3"],  # 惊吓/嘶嘶
        "growl": [],                                  # 低吼（暂空，hiss 兜底）
    }
    
    for name, files in sound_map.items():
        items = []
        for fname in files:
            path = os.path.join(cat_dir, fname)
            if os.path.exists(path):
                s = load_one(path)
                if s:
                    items.append((s, fname))
        if items:
            loaded[name] = items
            print(f"  ✅ {name}: {len(items)} sounds")
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
            hi_keywords = ["hi", "嗨", "嘿", "hello", "你好", "哈喽", "哈啰", "hey", "猫咪", "猫猫", "喵"]
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
    global palm_history, pet_cooldown_until, eye_pet_close_until, hiss_cooldown_until
    
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
            min_hand_detection_confidence=0.7,  # 提高阈值减少误检
            min_tracking_confidence=0.7,  # 提高跟踪置信度
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
                    
                    # 只在手掌位于画面下半部分时才记录（排除靠近脸部的误检）
                    if palm_cy > 0.35:
                        palm_history.append((palm_cx, palm_cy, now))
                    else:
                        palm_history.clear()  # 手在画面上部，可能是误检
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
                        
                        if total_changes >= SHAKE_MIN_COUNT and total_dist > SHAKE_THRESHOLD and len(palm_history) >= 15:
                            # 检测是否握拳（指尖靠近掌心）
                            palm_center = hand[0]  # 手腕
                            finger_tips = [hand[4], hand[8], hand[12], hand[16], hand[20]]
                            avg_tip_dist = sum(
                                math.sqrt((t.x - palm_center.x)**2 + (t.y - palm_center.y)**2)
                                for t in finger_tips
                            ) / len(finger_tips)
                            
                            is_fist = avg_tip_dist < FIST_THRESHOLD
                            
                            if is_fist and now > hiss_cooldown_until:
                                # 握拳摇晃 → 惊吓/低吼
                                print(f"[GESTURE] Fist shake! (dist={avg_tip_dist:.2f})", flush=True)
                                if not play_cat_sound("hiss", 0.5, "🖐️握拳"):
                                    play_cat_sound("growl", 0.5, "🖐️握拳")
                                hiss_cooldown_until = now + 2.0
                                palm_history.clear()
                                cv2.putText(debug, "HISS!", (120, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)
                            elif not is_fist and now > pet_cooldown_until:
                                # 张手摇晃 → pet 抚摸
                                print(f"[GESTURE] Pet! (open hand)", flush=True)
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

def draw_cat_face(screen, eye_open_ratio=1.0):
    """画猫咪脸 v5：圆脸 + 大耳 + 猫眼 + 鼻子 + 嘴 + 胡须 + 尾巴"""
    global _breath_t, _whisker_t, _tail_t, _ear_tilt, _ear_target
    _breath_t += 0.03
    _whisker_t += 0.04
    _tail_t += 0.02
    breath_scale = 1.0 + 0.015 * math.sin(_breath_t * 1.5)

    cx = WIDTH // 2
    face_y = EYE_Y + 25

    # ── 耳朵倾斜动画（偶尔动一下）──
    for i in range(2):
        if random.random() < 0.005:
            _ear_target[i] = random.uniform(-8, 8)
        _ear_tilt[i] += (_ear_target[i] - _ear_tilt[i]) * 0.05

    # ── 耳朵（大三角，带绒毛）──
    for idx, side in enumerate([-1, 1]):
        ear_cx = cx + side * 100
        tilt = _ear_tilt[idx]
        pts_outer = [
            (ear_cx + side * 5 + tilt, EAR_Y),
            (ear_cx - EAR_W - 3, EAR_Y + EAR_H + 8),
            (ear_cx + EAR_W + 3, EAR_Y + EAR_H + 8),
        ]
        pygame.draw.polygon(screen, EAR_COLOR, pts_outer)
        iw = int(EAR_W * 0.55)
        pts_inner = [
            (ear_cx + side * 3 + tilt, EAR_Y + 14),
            (ear_cx - iw, EAR_Y + EAR_H + 2),
            (ear_cx + iw, EAR_Y + EAR_H + 2),
        ]
        pygame.draw.polygon(screen, EAR_INNER_COLOR, pts_inner)
        # 耳尖绒毛
        for j in range(3):
            fx = ear_cx + side * 2 + tilt + random.randint(-3, 3)
            fy = EAR_Y + 2 + j * 3
            pygame.draw.line(screen, EAR_TUFT_COLOR, (fx, fy), (fx + side * 4, fy - 5), 1)

    # ── 脸型（椭圆 + 底部阴影）──
    face_rx, face_ry = 125, 110
    face_rect = pygame.Rect(0, 0, face_rx * 2, face_ry * 2)
    face_rect.center = (cx, face_y)
    pygame.draw.ellipse(screen, FACE_COLOR, face_rect)
    # 底部渐变阴影
    shadow_surf = pygame.Surface((face_rx * 2, face_ry), pygame.SRCALPHA)
    for row in range(face_ry):
        alpha = int(35 * (row / face_ry))
        pygame.draw.line(shadow_surf, (180, 140, 100, alpha), (0, row), (face_rx * 2, row))
    screen.blit(shadow_surf, (cx - face_rx, face_y))
    pygame.draw.ellipse(screen, (235, 200, 160), face_rect, 2)

    # ── 额头 M 形花纹 ──
    mark_color = (235, 200, 160)
    mark_y = face_y - face_ry + 35
    pygame.draw.lines(screen, mark_color, False,
                      [(cx - 5, mark_y + 18), (cx - 30, mark_y + 2), (cx - 58, mark_y + 20)], 3)
    pygame.draw.lines(screen, mark_color, False,
                      [(cx + 5, mark_y + 18), (cx + 30, mark_y + 2), (cx + 58, mark_y + 20)], 3)

    # ── 眼睛（大猫眼 + 绿色渐变虹膜 + 竖瞳）──
    for ex in [EYE_LEFT_X, EYE_RIGHT_X]:
        if eye_open_ratio < 0.2:
            draw_crescent_eye(screen, ex, EYE_Y, int(EYE_W * 0.9))
        else:
            w = int(EYE_W * breath_scale * 1.1)
            h = max(4, int(EYE_H * eye_open_ratio * breath_scale * 1.1))
            eye_rect = pygame.Rect(0, 0, w, h)
            eye_rect.center = (ex, EYE_Y)
            pygame.draw.ellipse(screen, EYE_COLOR, eye_rect)
            pygame.draw.ellipse(screen, (50, 50, 70), eye_rect, 2)

            if eye_open_ratio > 0.15:
                ix = ex + int(pupil_offset[0] * PUPIL_TRACK_RANGE)
                iy = EYE_Y + int(pupil_offset[1] * PUPIL_TRACK_RANGE * 0.5)
                iris_w = int(w * 0.72)
                iris_h = int(h * 0.88)
                iris_rect = pygame.Rect(0, 0, iris_w, iris_h)
                iris_rect.center = (ix, iy)
                pygame.draw.ellipse(screen, IRIS_COLOR, iris_rect)
                # 虹膜内圈（更亮）
                inner_rect = pygame.Rect(0, 0, int(iris_w * 0.6), int(iris_h * 0.6))
                inner_rect.center = (ix, iy)
                pygame.draw.ellipse(screen, (100, 200, 100), inner_rect)
                # 竖瞳
                pw = max(3, int(iris_w * 0.18))
                ph = max(5, int(iris_h * 0.8 * eye_open_ratio))
                pupil_rect = pygame.Rect(0, 0, pw, ph)
                pupil_rect.center = (ix, iy)
                pygame.draw.ellipse(screen, PUPIL_COLOR, pupil_rect)
                # 大高光
                hx = ix - int(iris_w * 0.2)
                hy = iy - int(iris_h * 0.25)
                hr = max(4, int(iris_w * 0.15))
                pygame.draw.circle(screen, HIGHLIGHT_COLOR, (hx, hy), hr)
                # 小高光
                hx2 = ix + int(iris_w * 0.15)
                hy2 = iy + int(iris_h * 0.2)
                hr2 = max(2, int(iris_w * 0.08))
                pygame.draw.circle(screen, (255, 255, 255, 200), (hx2, hy2), hr2)

    # ── 腮红（被摸后短暂显示，带淡出）──
    if time.time() < _blush_until:
        remaining = _blush_until - time.time()
        blush_alpha = int(120 * min(1, remaining))
        blush_y = EYE_Y + 48
        blush_w = int(32 * breath_scale)
        blush_h = int(16 * breath_scale)
        blush_surf = pygame.Surface((blush_w, blush_h), pygame.SRCALPHA)
        pygame.draw.ellipse(blush_surf, (255, 150, 160, blush_alpha), (0, 0, blush_w, blush_h))
        screen.blit(blush_surf, (EYE_LEFT_X - 55, blush_y))
        screen.blit(blush_surf, (EYE_RIGHT_X + 23, blush_y))

    # ── 鼻子（粉色倒三角 + 光泽）──
    nose_pts = [
        (cx, NOSE_Y - NOSE_SIZE // 2 + 2),
        (cx - NOSE_SIZE + 2, NOSE_Y + NOSE_SIZE // 2),
        (cx + NOSE_SIZE - 2, NOSE_Y + NOSE_SIZE // 2),
    ]
    pygame.draw.polygon(screen, NOSE_COLOR, nose_pts)
    pygame.draw.ellipse(screen, (255, 190, 195), (cx - 4, NOSE_Y - 3, 6, 4))

    # ── 嘴巴线 ──
    pygame.draw.line(screen, MOUTH_COLOR, (cx, NOSE_Y + NOSE_SIZE // 2 - 1), (cx, MOUTH_Y), 2)

    # ── 胡须（3 根每侧，呼吸微动）──
    whisker_breath = math.sin(_whisker_t) * 2
    for side in [-1, 1]:
        for i, dy in enumerate([-8, 0, 8]):
            wx_start = cx + side * 28
            wy_start = NOSE_Y + 12 + dy
            wx_end = cx + side * 118
            wy_end = NOSE_Y + 6 + dy + (i - 1) * 5 + whisker_breath
            pygame.draw.line(screen, WHISKER_COLOR, (wx_start, wy_start), (wx_end, wy_end), 2)
            mid_x = (wx_start + wx_end) // 2
            mid_y = (wy_start + wy_end) // 2
            pygame.draw.line(screen, (220, 220, 220), (mid_x, mid_y), (wx_end, wy_end), 1)

    # ── 尾巴（右侧摆动）──
    tail_cx = cx + face_rx + 10
    tail_y = face_y + 40
    tail_swing = math.sin(_tail_t * 2) * 15
    tail_pts = [
        (tail_cx, tail_y),
        (tail_cx + 20, tail_y - 30 + tail_swing * 0.3),
        (tail_cx + 35, tail_y - 55 + tail_swing * 0.7),
        (tail_cx + 25, tail_y - 75 + tail_swing),
    ]
    pygame.draw.lines(screen, TAIL_COLOR, False, tail_pts, 6)
    pygame.draw.circle(screen, (240, 190, 150), tail_pts[-1], 5)

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

# ── 绘制嘴巴（猫咪 W 形 + 舌头）──────────────────────────
_mouth_breathe = 0.0

def draw_mouth(screen, open_ratio):
    global _mouth_breathe
    _mouth_breathe += 0.02
    cx = WIDTH // 2

    if open_ratio < 0.15:
        # 闭嘴：W 形猫嘴
        # 中线（鼻下到嘴角）
        pygame.draw.line(screen, MOUTH_COLOR, (cx, NOSE_Y + 8), (cx, MOUTH_Y + 3), 2)
        # 左弧
        pts_l = []
        for i in range(12):
            t = i / 11.0
            x = cx - t * MOUTH_W
            y = MOUTH_Y + 3 + math.sin(t * math.pi) * 6
            pts_l.append((x, y))
        pygame.draw.lines(screen, MOUTH_COLOR, False, pts_l, 2)
        # 右弧
        pts_r = []
        for i in range(12):
            t = i / 11.0
            x = cx + t * MOUTH_W
            y = MOUTH_Y + 3 + math.sin(t * math.pi) * 6
            pts_r.append((x, y))
        pygame.draw.lines(screen, MOUTH_COLOR, False, pts_r, 2)
    else:
        # 张嘴：椭圆形
        w = int(MOUTH_W * (1.0 + 0.6 * open_ratio))
        h = int(MOUTH_H_CLOSED + (MOUTH_H_OPEN - MOUTH_H_CLOSED) * open_ratio)
        # 深色喉咙
        throat_rect = pygame.Rect(cx - w + 3, MOUTH_Y, (w - 3) * 2, h)
        pygame.draw.ellipse(screen, (80, 30, 40), throat_rect)
        # 嘴巴主体
        mouth_rect = pygame.Rect(cx - w, MOUTH_Y - 2, w * 2, h + 4)
        pygame.draw.ellipse(screen, MOUTH_COLOR, mouth_rect)
        # 舌头（圆润，伸出来）
        if open_ratio > 0.4:
            tw = int(w * 0.5)
            th = int(h * 0.4)
            ty = MOUTH_Y + int(h * 0.35)
            tongue_rect = pygame.Rect(cx - tw, ty, tw * 2, th)
            pygame.draw.ellipse(screen, TONGUE_COLOR, tongue_rect)
            # 舌头高光
            shine_rect = pygame.Rect(cx - tw + 3, ty + 2, tw - 3, th // 3)
            pygame.draw.ellipse(screen, (255, 170, 170), shine_rect)
        # 中线
        pygame.draw.line(screen, MOUTH_COLOR, (cx, NOSE_Y + 8), (cx, MOUTH_Y), 2)

# ── 爱心粒子系统 ─────────────────────────────────────────
_hearts = []  # [(x, y, vx, vy, life, max_life), ...]
_blush_until = 0  # 腮红显示截止时间
_whisker_t = 0     # 胡须动画时间
_tail_t = 0        # 尾巴动画时间
_ear_tilt = [0, 0] # 耳朵倾斜角度 [左, 右]
_ear_target = [0, 0]
_sleep_zzz = []    # [(x, y, life, max_life), ...]
_pupil_dilation = 1.0  # 瞳孔缩放

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
# ── 猫爪（带肉垫细节）────────────────────────────────────────
def draw_feet(screen):
    """底部猫爪 + 肉垫"""
    paw_y = HEIGHT - 50
    for px in [WIDTH // 2 - 55, WIDTH // 2 + 55]:
        # 掌心（椭圆）
        paw_rect = pygame.Rect(0, 0, 34, 24)
        paw_rect.center = (px, paw_y)
        pygame.draw.ellipse(screen, PAW_COLOR, paw_rect)
        # 肉垫（大椭圆）
        pad_rect = pygame.Rect(0, 0, 14, 11)
        pad_rect.center = (px, paw_y + 3)
        pygame.draw.ellipse(screen, PAW_PAD_COLOR, pad_rect)
        # 小爪尖（3 个圆）
        for dx in [-11, 0, 11]:
            tip_rect = pygame.Rect(0, 0, 9, 9)
            tip_rect.center = (px + dx, paw_y - 11)
            pygame.draw.ellipse(screen, PAW_COLOR, tip_rect)
        # 爪尖肉垫
        for dx in [-11, 0, 11]:
            tip_pad = pygame.Rect(0, 0, 4, 4)
            tip_pad.center = (px + dx, paw_y - 10)
            pygame.draw.ellipse(screen, PAW_PAD_COLOR, tip_pad)

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
    global voice_wake_until, voice_event_type, voice_cooldown_until, next_random_sound
    
    global sound_enabled, sounds
    pygame.init()
    sound_enabled = False
    sounds = {}
    try:
        pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2)
        sound_enabled = True
    except pygame.error as e:
        print(f"[WARN] 音频失败: {e}")
    
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("🐱 Cat Pet")
    try:
        import ctypes
        hwnd = pygame.display.get_wm_info()["window"]
        ctypes.windll.user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, 0x0001 | 0x0002)
    except Exception:
        pass
    
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 14)
    
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
            eye_open += ((SNOOZE_EYE_H / EYE_H) - eye_open) * min(1, dt * 1.5)
            # 睡觉时播放打哈欠音效 + ZZZ
            if not hasattr(main, '_sleep_sound_played') or not main._sleep_sound_played:
                play_cat_sound("purr", 0.3, "😴睡觉")
                main._sleep_sound_played = True
            # 持续生成 ZZZ
            if random.random() < 0.02:
                spawn_zzz()
        else:
            eye_open += (1.0 - eye_open) * min(1, dt * 3)
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
            play_cat_sound("purr", 0.5, "🖐️抚摸")
            spawn_hearts(6)
            _blush_until = now + 3.0
            eye_pet_close_until = now - 1
        
        # 语音唤醒发声
        if now < voice_wake_until and now > voice_wake_until - 1.9 and sound_enabled and now > voice_cooldown_until:
            if voice_event_type == "hi":
                play_cat_sound("meow", 0.6, "🎙️打招呼")
            else:
                play_cat_sound("meow", 0.5, "🎙️呼唤")
            voice_wake_until = now - 1
            voice_event_type = None
            voice_cooldown_until = now + 3.0
        
        # 随机声音
        next_random_sound -= dt
        if next_random_sound <= 0 and sound_enabled:
            pool = random.choice(["meow", "meow", "purr", "purr"])
            print(f"[DEBUG] random timer fired, trying {pool}", flush=True)
            play_cat_sound(pool, random.uniform(0.2, 0.4), "🎲随机")
            next_random_sound = random.uniform(*RANDOM_SOUND_INTERVAL)
        
        # 绘制（渐变背景）
        for row in range(HEIGHT):
            t = row / HEIGHT
            r = int(30 + 15 * t)
            g = int(30 + 10 * t)
            b = int(40 + 20 * t)
            pygame.draw.line(screen, (r, g, b), (0, row), (WIDTH, row))
        draw_feet(screen)
        draw_cat_face(screen, eye_open)
        draw_mouth(screen, mouth_open)
        draw_hearts(screen, dt)
        if state == "sleeping":
            draw_zzz(screen, dt)
        draw_state(screen, font, state, clock.get_fps())
        pygame.display.flip()
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
