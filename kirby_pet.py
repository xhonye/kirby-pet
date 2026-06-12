"""
Kirby Desktop Pet — MVP
- pygame 窗口，深色背景
- 卡比风格大眼睛，跟随人脸方向
- MediaPipe 人脸检测（后台线程）
- 随机眨眼、打瞌睡
- 随机哼唧声（合成音）
"""

import pygame
import cv2
import mediapipe as mp
import numpy as np
import threading
import time
import random
import math
import sys

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

# 眨眼
BLINK_INTERVAL = (2, 6)
BLINK_DURATION = 0.15

# 打瞌睡
SLEEP_TIMEOUT = 15
SNOOZE_EYE_H = 20

# 哼唧
HUM_INTERVAL = (4, 12)
HUM_FREQS = [440, 523, 659, 784, 880]
SAMPLE_RATE = 22050

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
next_hum = random.uniform(*HUM_INTERVAL)

# ── 生成哼唧音（纯 numpy，不写文件）────────────────────
def make_hum_sound(freq, duration=0.3):
    """生成正弦波+谐波叠加的哼唧音，返回 pygame.Sound"""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    wave_data = (
        0.5 * np.sin(2 * np.pi * freq * t) +
        0.3 * np.sin(2 * np.pi * freq * 1.5 * t) +
        0.2 * np.sin(2 * np.pi * freq * 2 * t)
    )
    # ADSR 简化
    env = np.ones_like(t)
    attack = int(0.05 * SAMPLE_RATE)
    decay = int(0.1 * SAMPLE_RATE)
    if attack > 0:
        env[:attack] = np.linspace(0, 1, attack)
    if decay > 0:
        env[-decay:] = np.linspace(1, 0, decay)
    wave_data *= env * 0.4
    # 转 int16，立体声
    samples = (wave_data * 32767).astype(np.int16)
    stereo = np.column_stack((samples, samples))
    sound = pygame.sndarray.make_sound(stereo)
    return sound

# ── MediaPipe 人脸检测线程（含调试窗口）──────────────
camera_status = "init"       # "init" / "opened" / "failed" / "no_face" / "tracking"
camera_debug_frame = None    # numpy array for debug window
detection_count = 0

def face_detection_loop():
    global face_x, face_y, last_face_time, camera_status
    global camera_debug_frame, detection_count
    
    mp_face = mp.solutions.face_detection
    mp_draw = mp.solutions.drawing_utils
    
    print("[INFO] 正在打开摄像头...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        camera_status = "failed"
        print("[ERROR] 摄像头未打开！检查是否被其他程序占用")
        return
    
    camera_status = "opened"
    print("[INFO] 摄像头已打开，开始人脸检测")
    print("[INFO] 按 Q 关闭调试窗口")
    
    with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.4) as fd:
        while True:
            ret, frame = cap.read()
            if not ret:
                camera_status = "failed"
                time.sleep(0.1)
                continue
            
            small = cv2.resize(frame, (320, 240))
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            result = fd.process(rgb)
            
            debug = small.copy()
            
            if result.detections:
                det = result.detections[0]
                bbox = det.location_data.relative_bounding_box
                cx = bbox.xmin + bbox.width / 2
                cy = bbox.ymin + bbox.height / 2
                face_x = cx
                face_y = cy
                last_face_time = time.time()
                detection_count += 1
                camera_status = "tracking"
                
                # 画检测框
                h, w, _ = debug.shape
                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                x2 = int((bbox.xmin + bbox.width) * w)
                y2 = int((bbox.ymin + bbox.height) * h)
                cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(debug, f"TRACKING", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(debug, f"face=({cx:.2f},{cy:.2f})", (5, 225),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            else:
                face_x = None
                face_y = None
                camera_status = "no_face"
                cv2.putText(debug, "NO FACE", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            camera_debug_frame = debug
            
            # 调试窗口
            cv2.imshow("Kirby Debug - Camera", debug)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            time.sleep(0.03)

# ── 绘制眼睛 ────────────────────────────────────────────
def draw_eyes(screen, eye_open_ratio=1.0):
    for ex in [EYE_LEFT_X, EYE_RIGHT_X]:
        # 眼白
        h = max(3, int(EYE_H * eye_open_ratio))
        eye_rect = pygame.Rect(0, 0, EYE_W, h)
        eye_rect.center = (ex, EYE_Y)
        pygame.draw.ellipse(screen, EYE_COLOR, eye_rect)
        pygame.draw.ellipse(screen, (180, 140, 160), eye_rect, 2)
        
        if eye_open_ratio > 0.3:
            # 瞳孔
            px = ex + int(pupil_offset[0] * PUPIL_TRACK_RANGE)
            py = EYE_Y + int(pupil_offset[1] * PUPIL_TRACK_RANGE * 0.5)
            pr = max(3, int(PUPIL_R * eye_open_ratio))
            pygame.draw.circle(screen, PUPIL_COLOR, (px, py), pr)
            
            # 高光
            hx = px - int(pr * 0.3)
            hy = py - int(pr * 0.3)
            hr = max(3, int(pr * 0.35))
            pygame.draw.circle(screen, (255, 255, 255), (hx, hy), hr)
    
    # 腮红
    blush_y = EYE_Y + 50
    blush_surf = pygame.Surface((60, 30), pygame.SRCALPHA)
    pygame.draw.ellipse(blush_surf, BLUSH_COLOR, (0, 0, 60, 30))
    screen.blit(blush_surf, (EYE_LEFT_X - 80, blush_y))
    screen.blit(blush_surf, (EYE_RIGHT_X + 20, blush_y))

# ── 状态文字 ────────────────────────────────────────────
def draw_state(screen, font, state_text, fps_val):
    # 顶部状态
    text = font.render(f"{state_text}  FPS:{fps_val:.0f}", True, (100, 100, 120))
    screen.blit(text, (10, 10))
    
    # 底部摄像头状态
    cam_colors = {
        "init": (180, 180, 100),   # 黄
        "opened": (100, 180, 100), # 浅绿
        "failed": (255, 80, 80),   # 红
        "no_face": (180, 130, 80), # 橙
        "tracking": (80, 220, 80), # 绿
    }
    cam_labels = {
        "init": "CAM: initializing...",
        "opened": "CAM: opened, waiting...",
        "failed": "CAM: FAILED (check camera)",
        "no_face": "CAM: no face detected",
        "tracking": f"CAM: tracking (#{detection_count})",
    }
    cam_color = cam_colors.get(camera_status, (100, 100, 100))
    cam_label = cam_labels.get(camera_status, f"CAM: {camera_status}")
    cam_text = font.render(cam_label, True, cam_color)
    screen.blit(cam_text, (10, HEIGHT - 25))

# ── 主循环 ──────────────────────────────────────────────
def main():
    global state, pupil_offset, target_pupil
    global blink_timer, is_blinking, blink_end, next_blink
    global next_hum, wander_timer
    
    pygame.init()
    sound_enabled = False
    try:
        pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2)
        sound_enabled = True
    except pygame.error as e:
        print(f"[WARN] 音频初始化失败: {e}, 静音模式运行")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Kirby Pet")
    # 窗口置顶（Windows）
    try:
        import ctypes
        hwnd = pygame.display.get_wm_info()["window"]
        ctypes.windll.user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, 0x0001 | 0x0002)
    except Exception:
        pass
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 14)
    
    # 生成哼唧音效
    hum_sounds = []
    if sound_enabled:
        for f in HUM_FREQS:
            hum_sounds.append(make_hum_sound(f, duration=random.uniform(0.2, 0.5)))
    
    # 启动人脸检测线程
    t = threading.Thread(target=face_detection_loop, daemon=True)
    t.start()
    
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
            time_since_face = now - wander_timer
            if time_since_face > SLEEP_TIMEOUT:
                state = "sleeping"
                sleep_level = min(1.0, sleep_level + dt * 0.3)
            else:
                state = "idle"
                if random.random() < 0.02:
                    target_pupil[0] = random.uniform(-0.8, 0.8)
                    target_pupil[1] = random.uniform(-0.3, 0.3)
        
        # 瞳孔平滑
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
        if is_blinking:
            eye_open = max(0.05, eye_open - dt * 15)
        elif state == "sleeping":
            target = SNOOZE_EYE_H / EYE_H
            eye_open += (target - eye_open) * dt * 2
        else:
            eye_open += (1.0 - eye_open) * dt * 10
        
        # 随机哼唧
        next_hum -= dt
        if next_hum <= 0 and hum_sounds:
            s = random.choice(hum_sounds)
            s.set_volume(random.uniform(0.2, 0.5))
            s.play()
            next_hum = random.uniform(*HUM_INTERVAL)
        
        # 绘制
        screen.fill(BG_COLOR)
        draw_eyes(screen, eye_open)
        draw_state(screen, font, state, clock.get_fps())
        pygame.display.flip()
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
