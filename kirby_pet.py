"""
Kirby Desktop Pet — v2
- pygame 窗口，深色背景
- 卡比风格大眼睛 + 嘴巴，跟随人脸方向
- MediaPipe 人脸检测（后台线程）
- MediaPipe 手部检测：摸头时闭眼+发声
- 语音识别：喊"卡比"时张嘴回应
- 随机眨眼、打瞌睡、随机卡比声音
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

# 眼睛参数
EYE_COLOR = (255, 200, 220)
PUPIL_COLOR = (30, 30, 80)
BLUSH_COLOR = (255, 130, 150, 180)
EYE_W, EYE_H = 140, 160
PUPIL_R = 28
EYE_SPACING = 80
EYE_Y = HEIGHT // 2 - 20
EYE_LEFT_X = WIDTH // 2 - EYE_SPACING
EYE_RIGHT_X = WIDTH // 2 + EYE_SPACING
PUPIL_TRACK_RANGE = 30

# 嘴巴参数
MOUTH_Y = EYE_Y + 100
MOUTH_W = 40
MOUTH_H_CLOSED = 6
MOUTH_H_OPEN = 35

# 眨眼
BLINK_INTERVAL = (2, 6)
BLINK_DURATION = 0.15

# 打瞌睡
SLEEP_TIMEOUT = 15
SNOOZE_EYE_H = 20

# 声音
SAMPLE_RATE = 22050
RANDOM_SOUND_INTERVAL = (6, 18)

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
mouth_open = 0.0          # 0.0=闭, 1.0=张开
mouth_target = 0.0
mouth_event_until = 0     # 张嘴事件截止时间

# 手部检测状态
hand_near = False         # 手是否在卡比头上
hand_pet_until = 0        # 被摸的反应持续时间

# 语音唤醒状态
voice_wake_until = 0      # 被喊名字的反应持续时间

# 下一次随机声音
next_random_sound = random.uniform(*RANDOM_SOUND_INTERVAL)

# 摄像头
camera_status = "init"
camera_debug_frame = None
detection_count = 0

# ── 音效加载 ────────────────────────────────────────────
def load_sounds():
    """从 sounds/ 目录加载 MP3 音效文件"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sounds_dir = os.path.join(script_dir, "sounds")
    loaded = {}
    
    sound_map = {
        "poyo": ["poyo.wav", "poyo.mp3"],
        "happy": ["happy.wav", "happy.mp3"],
        "inhale": ["inhale.wav", "inhale.mp3"],
        "hurt": ["hurt.wav", "hurt.mp3"],
        "pet": ["pet.wav", "pet.mp3"],
    }
    
    for name, files in sound_map.items():
        for fname in files:
            path = os.path.join(sounds_dir, fname)
            if os.path.exists(path):
                try:
                    loaded[name] = [pygame.mixer.Sound(path)]
                    print(f"  ✅ {name}: {fname}")
                except Exception as e:
                    print(f"  ❌ {name}: {e}")
            else:
                print(f"  ⚠️ {name}: {fname} not found")
    
    return loaded

# ── 语音识别线程 ────────────────────────────────────────
def voice_recognition_loop():
    """监听麦克风，检测到喊"卡比"时触发回应"""
    global voice_wake_until, mouth_event_until
    
    try:
        import speech_recognition as sr
    except ImportError:
        print("[WARN] speech_recognition 未安装，语音功能禁用")
        print("[WARN] 运行: pip install SpeechRecognition")
        return
    
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    
    print('[INFO] 语音识别启动，喊"卡比"试试！')
    
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
    
    while True:
        try:
            with mic as source:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=3)
            
            # 用 Google 免费 API 识别中文
            text = recognizer.recognize_google(audio, language="zh-CN")
            print(f"[VOICE] 识别到: {text}")
            
            if '卡比' in text or '卡逼' in text or '科比' in text:
                print('[VOICE] ★ 检测到呼唤卡比！')
                voice_wake_until = time.time() + 2.0
                mouth_event_until = time.time() + 1.5
                
        except sr.WaitTimeoutError:
            pass
        except sr.UnknownValueError:
            pass
        except sr.RequestError as e:
            print(f"[WARN] 语音服务错误: {e}")
            time.sleep(3)
        except Exception as e:
            print(f"[WARN] 语音线程异常: {e}")
            time.sleep(1)

# ── MediaPipe 人脸+手部检测线程 ──────────────────────────
def camera_detection_loop():
    global face_x, face_y, last_face_time, camera_status
    global camera_debug_frame, detection_count
    global hand_near, hand_pet_until
    
    import mediapipe as mp_lib
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    face_model = os.path.join(script_dir, "models", "blaze_face_short_range.tflite")
    hand_model = os.path.join(script_dir, "models", "hand_landmarker.task")
    
    if not os.path.exists(face_model):
        camera_status = "failed"
        print(f"[ERROR] 人脸模型不存在: {face_model}")
        return
    
    # 人脸检测器
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
            min_hand_detection_confidence=0.4,
            running_mode=vision.RunningMode.VIDEO,
        )
        hand_detector = vision.HandLandmarker.create_from_options(hand_opts)
        print("[INFO] 手部检测已加载")
    else:
        print("[WARN] 手部模型不存在，手部检测禁用")
    
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
            face_x = cx
            face_y = cy
            last_face_time = now
            detection_count += 1
            camera_status = "tracking"
            
            x1, y1 = bbox.origin_x, bbox.origin_y
            x2, y2 = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
            cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(debug, "FACE", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            # 手部检测（只在检测到脸时才检测手）
            if hand_detector is not None:
                hand_result = hand_detector.detect_for_video(mp_image, ts)
                if hand_result.hand_landmarks:
                    hand = hand_result.hand_landmarks[0]
                    # 手掌中心（手腕 landmark 0）
                    hx = hand[0].x
                    hy = hand[0].y
                    
                    # 画手部关键点
                    for lm in hand:
                        px, py = int(lm.x * 320), int(lm.y * 240)
                        cv2.circle(debug, (px, py), 2, (255, 200, 0), -1)
                    
                    # 判断手是否靠近脸部（距离 < 0.3）
                    dist = math.sqrt((hx - cx)**2 + (hy - cy)**2)
                    if dist < 0.35:
                        if not hand_near:
                            print(f"[TOUCH] 手摸到了卡比！距离={dist:.2f}")
                            hand_pet_until = now + 1.5
                        hand_near = True
                        cv2.putText(debug, "PET!", (int(hx*320), int(hy*240)-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                    else:
                        hand_near = False
                else:
                    hand_near = False
        else:
            face_x = None
            face_y = None
            camera_status = "no_face"
            hand_near = False
            cv2.putText(debug, "NO FACE", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        camera_debug_frame = debug
        cv2.imshow("Kirby Debug - Camera", debug)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(0.03)

# ── 绘制嘴巴 ────────────────────────────────────────────
def draw_mouth(screen, open_ratio):
    """open_ratio: 0.0=闭嘴微笑, 1.0=大张嘴"""
    center_x = WIDTH // 2
    
    if open_ratio < 0.15:
        # 闭嘴：微笑弧线
        points = []
        for i in range(20):
            t = i / 19.0
            x = center_x - MOUTH_W + t * MOUTH_W * 2
            y = MOUTH_Y + math.sin(t * math.pi) * 8
            points.append((x, y))
        if len(points) >= 2:
            pygame.draw.lines(screen, (200, 100, 120), False, points, 3)
    else:
        # 张嘴：椭圆
        h = int(MOUTH_H_CLOSED + (MOUTH_H_OPEN - MOUTH_H_CLOSED) * open_ratio)
        w = int(MOUTH_W * (0.8 + 0.2 * open_ratio))
        mouth_rect = pygame.Rect(0, 0, w * 2, h)
        mouth_rect.center = (center_x, MOUTH_Y)
        # 嘴巴内部（深色）
        pygame.draw.ellipse(screen, (120, 40, 50), mouth_rect)
        # 舌头（如果张得够大）
        if open_ratio > 0.5:
            tongue_rect = pygame.Rect(0, 0, int(w * 0.8), int(h * 0.4))
            tongue_rect.center = (center_x, MOUTH_Y + int(h * 0.2))
            pygame.draw.ellipse(screen, (220, 100, 100), tongue_rect)

# ── 绘制眼睛 ────────────────────────────────────────────
def draw_eyes(screen, eye_open_ratio=1.0):
    for ex in [EYE_LEFT_X, EYE_RIGHT_X]:
        h = max(3, int(EYE_H * eye_open_ratio))
        eye_rect = pygame.Rect(0, 0, EYE_W, h)
        eye_rect.center = (ex, EYE_Y)
        pygame.draw.ellipse(screen, EYE_COLOR, eye_rect)
        pygame.draw.ellipse(screen, (180, 140, 160), eye_rect, 2)
        
        if eye_open_ratio > 0.3:
            px = ex + int(pupil_offset[0] * PUPIL_TRACK_RANGE)
            py = EYE_Y + int(pupil_offset[1] * PUPIL_TRACK_RANGE * 0.5)
            pr = max(3, int(PUPIL_R * eye_open_ratio))
            pygame.draw.circle(screen, PUPIL_COLOR, (px, py), pr)
            
            hx = px - int(pr * 0.3)
            hy = py - int(pr * 0.3)
            hr = max(3, int(pr * 0.35))
            pygame.draw.circle(screen, (255, 255, 255), (hx, hy), hr)
    
    blush_y = EYE_Y + 50
    blush_surf = pygame.Surface((60, 30), pygame.SRCALPHA)
    pygame.draw.ellipse(blush_surf, BLUSH_COLOR, (0, 0, 60, 30))
    screen.blit(blush_surf, (EYE_LEFT_X - 80, blush_y))
    screen.blit(blush_surf, (EYE_RIGHT_X + 20, blush_y))

# ── 状态绘制 ────────────────────────────────────────────
def draw_state(screen, font, state_text, fps_val):
    text = font.render(f"{state_text}  FPS:{fps_val:.0f}", True, (100, 100, 120))
    screen.blit(text, (10, 10))
    
    cam_colors = {
        "init": (180, 180, 100),
        "opened": (100, 180, 100),
        "failed": (255, 80, 80),
        "no_face": (180, 130, 80),
        "tracking": (80, 220, 80),
    }
    cam_labels = {
        "init": "CAM: init...",
        "opened": "CAM: waiting...",
        "failed": "CAM: FAILED",
        "no_face": "CAM: no face",
        "tracking": f"FACE #{detection_count}" + (" | HAND!" if hand_near else ""),
    }
    cam_color = cam_colors.get(camera_status, (100, 100, 100))
    cam_label = cam_labels.get(camera_status, f"CAM: {camera_status}")
    cam_text = font.render(cam_label, True, cam_color)
    screen.blit(cam_text, (10, HEIGHT - 25))

# ── 主循环 ──────────────────────────────────────────────
def main():
    global state, pupil_offset, target_pupil
    global blink_timer, is_blinking, blink_end, next_blink
    global wander_timer, mouth_open, mouth_target
    global mouth_event_until, hand_near, hand_pet_until
    global voice_wake_until, next_random_sound
    
    pygame.init()
    sound_enabled = False
    try:
        pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2)
        sound_enabled = True
    except pygame.error as e:
        print(f"[WARN] 音频初始化失败: {e}, 静音模式运行")
    
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Kirby Pet")
    try:
        import ctypes
        hwnd = pygame.display.get_wm_info()["window"]
        ctypes.windll.user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, 0x0001 | 0x0002)
    except Exception:
        pass
    
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 14)
    
    # 加载音效文件
    sounds = {}
    if sound_enabled:
        print("[INFO] 加载音效...")
        sounds = load_sounds()
        print(f"[INFO] 已加载 {len(sounds)} 个音效")
    
    # 启动摄像头线程
    t_cam = threading.Thread(target=camera_detection_loop, daemon=True)
    t_cam.start()
    
    # 启动语音线程
    t_voice = threading.Thread(target=voice_recognition_loop, daemon=True)
    t_voice.start()
    
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
        
        # ── 状态机 ──
        if face_x is not None:
            state = "tracking"
            sleep_level = 0
            target_pupil[0] = (0.5 - face_x) * 2
            if face_y is not None:
                target_pupil[1] = (face_y - 0.5) * 1.5
            wander_timer = now
        else:
            time_since_face = now - wander_timer
            if time_since_face > SLEEP_TIMEOUT:
                state = "sleeping"
                sleep_level = min(1.0, sleep_level + dt * 0.3)
            else:
                state = "idle"
                if random.random() < 0.02:
                    target_pupil[0] = random.uniform(-0.8, 0.8)
                    target_pupil[1] = random.uniform(-0.3, 0.3)
        
        # ── 瞳孔平滑 ──
        for i in range(2):
            pupil_offset[i] += (target_pupil[i] - pupil_offset[i]) * min(1, dt * 8)
        
        # ── 眨眼 ──
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
        pet_eye_target = 0.4 if (hand_near or now < hand_pet_until) else None
        
        if is_blinking:
            eye_open = max(0.05, eye_open - dt * 15)
        elif pet_eye_target is not None:
            eye_open += (pet_eye_target - eye_open) * dt * 8
        elif state == "sleeping":
            target = SNOOZE_EYE_H / EYE_H
            eye_open += (target - eye_open) * dt * 2
        else:
            eye_open += (1.0 - eye_open) * dt * 10
        
        # ── 嘴巴状态 ──
        mouth_target = 0.0
        if now < mouth_event_until:
            mouth_target = 0.8  # 张嘴回应
        elif now < voice_wake_until:
            mouth_target = 0.5  # 听到名字微微张嘴
        
        mouth_open += (mouth_target - mouth_open) * min(1, dt * 12)
        
        # ── 手摸反应：发声 ──
        if hand_near and now < hand_pet_until and sound_enabled and sounds.get("pet"):
            if random.random() < 0.01:  # 不要每帧都播
                s = random.choice(sounds["pet"])
                s.set_volume(0.3)
                s.play()
                hand_pet_until = now + 1.5
        
        # ── 语音唤醒反应：发声 ──
        if now < voice_wake_until and now > voice_wake_until - 1.9 and sound_enabled:
            if sounds.get("poyo"):
                s = random.choice(sounds["poyo"])
                s.set_volume(0.5)
                s.play()
                voice_wake_until = now - 1  # 只播一次
        
        # ── 随机声音 ──
        next_random_sound -= dt
        if next_random_sound <= 0 and sound_enabled:
            pool = random.choice(["poyo", "happy", "inhale"])
            if sounds.get(pool):
                s = random.choice(sounds[pool])
                s.set_volume(random.uniform(0.15, 0.35))
                s.play()
            next_random_sound = random.uniform(*RANDOM_SOUND_INTERVAL)
        
        # ── 绘制 ──
        screen.fill(BG_COLOR)
        draw_eyes(screen, eye_open)
        draw_mouth(screen, mouth_open)
        draw_state(screen, font, state, clock.get_fps())
        pygame.display.flip()
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
