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

# 手势检测
PINCH_THRESHOLD = 0.06           # 捏合距离阈值（归一化坐标）
RELEASE_THRESHOLD = 0.12         # 释放距离阈值
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
pinching = False                 # 当前是否在捏合
pet_cooldown_until = 0           # pet 冷却
eye_pet_close_until = 0          # pet 时闭眼

# 语音唤醒状态
voice_wake_until = 0

# 下一次随机声音
next_random_sound = random.uniform(*RANDOM_SOUND_INTERVAL)

# 摄像头
camera_status = "init"
camera_debug_frame = None
detection_count = 0

# ── 音效加载 ────────────────────────────────────────────
def load_sounds():
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
                    break
                except Exception as e:
                    print(f"  ❌ {name}: {e}")
    return loaded

# ── 语音识别线程 ────────────────────────────────────────
def voice_recognition_loop():
    global voice_wake_until, mouth_event_until
    try:
        import speech_recognition as sr
    except ImportError:
        print("[WARN] speech_recognition 未安装，语音功能禁用")
        return
    
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    print("[INFO] 语音识别启动，喊卡比试试！")
    
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
    
    while True:
        try:
            with mic as source:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=3)
            text = recognizer.recognize_google(audio, language="zh-CN")
            print(f"[VOICE] {text}")
            if "卡比" in text or "卡逼" in text or "科比" in text:
                print("[VOICE] ★ 呼唤卡比！")
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
            print(f"[WARN] 语音异常: {e}")
            time.sleep(1)

# ── 摄像头检测线程（人脸+手势）──────────────────────────
def camera_detection_loop():
    global face_x, face_y, last_face_time, camera_status
    global camera_debug_frame, detection_count
    global pinching, pet_cooldown_until, eye_pet_close_until
    
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
    
    was_pinching = False
    
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
            
            # 手势检测：捏合→展开 = pet
            if hand_detector is not None:
                hand_result = hand_detector.detect_for_video(mp_image, ts)
                if hand_result.hand_landmarks:
                    hand = hand_result.hand_landmarks[0]
                    # 拇指尖(4) 和 食指尖(8)
                    thumb_tip = hand[4]
                    index_tip = hand[8]
                    dist = math.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
                    
                    # 画指尖连线
                    tx, ty = int(thumb_tip.x * 320), int(thumb_tip.y * 240)
                    ix, iy = int(index_tip.x * 320), int(index_tip.y * 240)
                    cv2.circle(debug, (tx, ty), 4, (255, 200, 0), -1)
                    cv2.circle(debug, (ix, iy), 4, (255, 200, 0), -1)
                    cv2.line(debug, (tx, ty), (ix, iy), (255, 200, 0), 1)
                            pinch_label = "PINCH" if dist < PINCH_THRESHOLD else f"pinch:{dist:.2f}"
                    cv2.putText(debug, pinch_label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255) if dist < PINCH_THRESHOLD else (200, 200, 200), 1)
                    
                    # 捏合状态机
                    if dist < PINCH_THRESHOLD:
                        if not was_pinching:
                            cv2.putText(debug, "PINCH!", (tx, ty - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)
                        was_pinching = True
                    elif was_pinching and dist > RELEASE_THRESHOLD:
                        # 捏合后展开 = pet 手势
                        was_pinching = False
                        if now > pet_cooldown_until:
                            print("[GESTURE] ★ Pet! (pinch-release)")
                            eye_pet_close_until = now + 1.5
                            pet_cooldown_until = now + GESTURE_COOLDOWN
                            cv2.putText(debug, "PET!", (120, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    elif dist > RELEASE_THRESHOLD * 2:
                        was_pinching = False
                else:
                    was_pinching = False
        else:
            face_x, face_y = None, None
            camera_status = "no_face"
            was_pinching = False
            cv2.putText(debug, "NO FACE", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        camera_debug_frame = debug
        cv2.imshow("Kirby Debug - Camera", debug)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(0.03)

# ── 绘制卡比风格眼睛 ────────────────────────────────────
def draw_eyes(screen, eye_open_ratio=1.0):
    for ex in [EYE_LEFT_X, EYE_RIGHT_X]:
        # 眼睛外轮廓（白色椭圆）
        h = max(5, int(EYE_H * eye_open_ratio))
        eye_rect = pygame.Rect(0, 0, EYE_W, h)
        eye_rect.center = (ex, EYE_Y)
        pygame.draw.ellipse(screen, EYE_COLOR, eye_rect)
        pygame.draw.ellipse(screen, (200, 200, 220), eye_rect, 2)
        
        if eye_open_ratio > 0.25:
            # 蓝色虹膜（比眼白小一圈）
            iris_w = int(EYE_W * 0.65)
            iris_h = int(h * 0.7)
            iris_rect = pygame.Rect(0, 0, iris_w, iris_h)
            ix = ex + int(pupil_offset[0] * PUPIL_TRACK_RANGE)
            iy = EYE_Y + int(pupil_offset[1] * PUPIL_TRACK_RANGE * 0.5)
            iris_rect.center = (ix, iy)
            pygame.draw.ellipse(screen, IRIS_COLOR, iris_rect)
            
            # 深色瞳孔
            pr = max(3, int(iris_w * 0.35 * eye_open_ratio))
            pygame.draw.circle(screen, PUPIL_COLOR, (ix, iy), pr)
            
            # 高光（左上）
            hx = ix - int(pr * 0.5)
            hy = iy - int(pr * 0.5)
            hr = max(2, int(pr * 0.4))
            pygame.draw.circle(screen, HIGHLIGHT_COLOR, (hx, hy), hr)
    
    # 腮红
    blush_y = EYE_Y + 45
    blush_surf = pygame.Surface((50, 25), pygame.SRCALPHA)
    pygame.draw.ellipse(blush_surf, BLUSH_COLOR, (0, 0, 50, 25))
    screen.blit(blush_surf, (EYE_LEFT_X - 70, blush_y))
    screen.blit(blush_surf, (EYE_RIGHT_X + 20, blush_y))

# ── 绘制倒三角嘴巴 ────────────────────────────────────
def draw_mouth(screen, open_ratio):
    cx = WIDTH // 2
    
    if open_ratio < 0.15:
        # 闭嘴：小倒三角（微笑）
        pts = [
            (cx - MOUTH_W, MOUTH_Y - 4),
            (cx + MOUTH_W, MOUTH_Y - 4),
            (cx, MOUTH_Y + MOUTH_H_CLOSED),
        ]
        pygame.draw.polygon(screen, MOUTH_COLOR, pts)
    else:
        # 张嘴：大倒三角
        w = int(MOUTH_W * (1.0 + 0.5 * open_ratio))
        h = int(MOUTH_H_CLOSED + (MOUTH_H_OPEN - MOUTH_H_CLOSED) * open_ratio)
        pts = [
            (cx - w, MOUTH_Y - 4),
            (cx + w, MOUTH_Y - 4),
            (cx, MOUTH_Y + h),
        ]
        pygame.draw.polygon(screen, MOUTH_COLOR, pts)
        
        # 舌头（张得够大时）
        if open_ratio > 0.4:
            tw = int(w * 0.5)
            th = int(h * 0.4)
            ty = MOUTH_Y + int(h * 0.4)
            tongue_pts = [
                (cx - tw, ty),
                (cx + tw, ty),
                (cx, ty + th),
            ]
            pygame.draw.polygon(screen, TONGUE_COLOR, tongue_pts)

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
    global voice_wake_until, next_random_sound
    
    pygame.init()
    sound_enabled = False
    try:
        pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2)
        sound_enabled = True
    except pygame.error as e:
        print(f"[WARN] 音频失败: {e}")
    
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
        else:
            eye_open += (1.0 - eye_open) * dt * 10
        
        # 嘴巴
        mouth_target = 0.0
        if now < mouth_event_until:
            mouth_target = 0.8
        elif now < voice_wake_until:
            mouth_target = 0.5
        mouth_open += (mouth_target - mouth_open) * min(1, dt * 12)
        
        # pet 时发声
        if now < eye_pet_close_until and now > eye_pet_close_until - 1.4 and sound_enabled:
            if sounds.get("pet"):
                s = random.choice(sounds["pet"])
                s.set_volume(0.35)
                s.play()
                eye_pet_close_until = now - 1  # 只播一次
        
        # 语音唤醒发声
        if now < voice_wake_until and now > voice_wake_until - 1.9 and sound_enabled:
            if sounds.get("poyo"):
                s = random.choice(sounds["poyo"])
                s.set_volume(0.5)
                s.play()
                voice_wake_until = now - 1
        
        # 随机声音
        next_random_sound -= dt
        if next_random_sound <= 0 and sound_enabled:
            pool = random.choice(["poyo", "happy", "inhale"])
            if sounds.get(pool):
                s = random.choice(sounds[pool])
                s.set_volume(random.uniform(0.15, 0.35))
                s.play()
            next_random_sound = random.uniform(*RANDOM_SOUND_INTERVAL)
        
        # 绘制
        screen.fill(BG_COLOR)
        draw_eyes(screen, eye_open)
        draw_mouth(screen, mouth_open)
        draw_state(screen, font, state, clock.get_fps())
        pygame.display.flip()
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
