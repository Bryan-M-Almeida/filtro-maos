from __future__ import annotations

import argparse
import ctypes
from collections import deque
import math
import random
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")
PALETTES = [(255, 110, 55), (90, 80, 255), (70, 240, 190), (245, 90, 220)]
ART_STYLES = ("POP ART USA", "CINEMA CLASSICO", "ART NOUVEAU EUROPA", "RETRATO REAL")


def ensure_model() -> Path:
    if MODEL_PATH.exists():
        return MODEL_PATH
    print("Baixando o modelo de detecção de mãos do MediaPipe...")
    request = urllib.request.Request(MODEL_URL, headers={"User-Agent": "maos-filter/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        MODEL_PATH.write_bytes(response.read())
    return MODEL_PATH


def open_camera(preferred_index: int):
    """Try the selected camera index first, then common Windows camera backends."""
    backends = [
        ("DirectShow", cv2.CAP_DSHOW),
        ("Media Foundation", cv2.CAP_MSMF),
        ("automático", cv2.CAP_ANY),
    ]
    indices = [preferred_index] + [i for i in range(6) if i != preferred_index]
    attempted = []
    for index in indices:
        for backend_name, backend in backends:
            attempted.append(f"{index}/{backend_name}")
            capture = cv2.VideoCapture(index, backend)
            if capture.isOpened():
                ok, frame = capture.read()
                if ok and frame is not None:
                    print(f"Câmera aberta: índice {index} ({backend_name}).")
                    return capture
            capture.release()
    details = ", ".join(attempted)
    raise SystemExit(
        "Não consegui abrir uma câmera. Confirme se o DroidCam está conectado e "
        "se a câmera virtual está habilitada no Windows. Tentativas: " + details + ". "
        "Tente iniciar com --camera 1, --camera 2 ou --camera 3."
    )


def distance(a, b) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def hand_openness(landmarks) -> float:
    """Approximate openness from how far the four fingertips extend from the palm."""
    wrist = landmarks[0]
    palm = distance(landmarks[9], wrist) + 1e-6
    tips = (8, 12, 16, 20)
    extended = sum(distance(landmarks[i], wrist) / palm > 1.35 for i in tips)
    return extended / 4.0


def is_l_gesture(landmarks) -> bool:
    """True when the index and thumb are extended while the other fingers are folded."""
    wrist = landmarks[0]
    index_extended = distance(landmarks[8], wrist) > distance(landmarks[6], wrist) * 1.18
    thumb_extended = distance(landmarks[4], landmarks[5]) > distance(landmarks[3], landmarks[5]) * 1.12
    other_fingers_folded = all(
        distance(landmarks[tip], wrist) < distance(landmarks[pip], wrist) * 1.12
        for tip, pip in ((12, 10), (16, 14), (20, 18))
    )
    return index_extended and thumb_extended and other_fingers_folded


def find_hand_frame(hands, width: int, height: int):
    """Build a polygon from the thumb/index tips of two L-shaped hands."""
    l_hands = [hand for hand in hands if is_l_gesture(hand)]
    if len(l_hands) != 2:
        return None
    points = np.array([
        [int(hand[i].x * width), int(hand[i].y * height)]
        for hand in l_hands for i in (4, 8)
    ], dtype=np.int32)
    hull = cv2.convexHull(points)
    x, y, box_width, box_height = cv2.boundingRect(hull)
    if box_width < 80 or box_height < 80 or cv2.contourArea(hull) < 5000:
        return None
    return hull.reshape(-1, 2)


def draw_hand(frame, landmarks, color) -> None:
    height, width = frame.shape[:2]
    points = [(int(p.x * width), int(p.y * height)) for p in landmarks]
    links = (
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
    )
    for start, end in links:
        cv2.line(frame, points[start], points[end], color, 2, cv2.LINE_AA)
    for point in points:
        cv2.circle(frame, point, 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, point, 6, color, 1, cv2.LINE_AA)


class ClapDetector:
    """Optional microphone listener. A detected clap advances the filter palette."""

    def __init__(self, threshold: float = 0.32):
        self.threshold = threshold
        self.last_clap = 0.0
        self.enabled = False
        self.stream = None

    def start(self):
        try:
            import sounddevice as sd

            def callback(indata, frames, _time, status):
                if status:
                    return
                level = float(np.sqrt(np.mean(np.square(indata))))
                now = time.monotonic()
                if level > self.threshold and now - self.last_clap > 0.8:
                    self.last_clap = now

            self.stream = sd.InputStream(channels=1, samplerate=16000, blocksize=1024,
                                         callback=callback)
            self.stream.start()
            self.enabled = True
        except Exception as exc:
            print(f"Microfone indisponível; o filtro continuará sem áudio ({exc}).")

    def stop(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()


def apply_comic_filter(frame):
    """Posterize the camera image, emphasize inked edges, and add a red/blue halftone tint."""
    smooth = cv2.bilateralFilter(frame, d=9, sigmaColor=95, sigmaSpace=95)
    quantized = (smooth // 32 * 32 + 16).clip(0, 255).astype(np.uint8)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.adaptiveThreshold(
        cv2.medianBlur(gray, 7), 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, 9, 5,
    )
    comic = cv2.bitwise_and(quantized, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))

    height, width = comic.shape[:2]
    split = np.zeros_like(comic)
    split[: height // 2] = (35, 40, 245)       # red upper half, BGR
    split[height // 2 :] = (245, 90, 35)       # blue lower half, BGR
    comic = cv2.addWeighted(comic, 0.84, split, 0.16, 0)

    # Small halftone dots, strongest in darker areas.
    yy, xx = np.indices((height, width))
    spacing = 9
    grid_x = (xx % spacing) - spacing / 2
    grid_y = (yy % spacing) - spacing / 2
    gray_float = gray.astype(np.float32)
    radius = (1.0 - gray_float / 255.0) * 3.2
    dots = (grid_x * grid_x + grid_y * grid_y) < (radius * radius)
    comic[dots] = (comic[dots].astype(np.float32) * 0.42).astype(np.uint8)
    return comic


def apply_art_filter(frame, style_index):
    """Apply one of four art-gallery looks to the camera image."""
    if style_index == 0:
        return apply_comic_filter(frame)

    height, width = frame.shape[:2]
    if style_index == 1:
        # Vintage American cinema: warm sepia, faded contrast, light film grain.
        sepia_matrix = np.array([
            [0.272, 0.534, 0.131],
            [0.349, 0.686, 0.168],
            [0.393, 0.769, 0.189],
        ], dtype=np.float32)
        styled = cv2.transform(frame, sepia_matrix)
        styled = cv2.convertScaleAbs(styled, alpha=0.88, beta=20)
        rng = np.random.default_rng(19)
        grain = rng.normal(0, 3.0, (height, width, 1)).astype(np.float32)
        styled = np.clip(styled.astype(np.float32) + grain, 0, 255).astype(np.uint8)
        return add_vignette(styled, 0.26)

    if style_index == 2:
        # European Art Nouveau: soft lines, muted greens, rose, and ivory.
        smooth = cv2.bilateralFilter(frame, d=9, sigmaColor=110, sigmaSpace=100)
        hsv = cv2.cvtColor(smooth, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= 0.68
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.08 + 12, 0, 255)
        pastel = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        wash = np.zeros_like(frame)
        for y in range(height):
            blend = y / max(height - 1, 1)
            wash[y, :] = (190 + int(28 * blend), 215 - int(20 * blend), 218 - int(18 * blend))
        styled = cv2.addWeighted(pastel, 0.82, wash, 0.18, 0)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 65, 130)
        styled[edges > 0] = (styled[edges > 0].astype(np.float32) * 0.72).astype(np.uint8)
        return styled

    # A softly painted, golden portrait treatment.
    smooth = cv2.bilateralFilter(frame, d=11, sigmaColor=125, sigmaSpace=115)
    warm = cv2.applyColorMap(cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_PINK)
    styled = cv2.addWeighted(smooth, 0.70, warm, 0.30, 8)
    return add_vignette(styled, 0.34)


def add_vignette(image, strength):
    height, width = image.shape[:2]
    y, x = np.ogrid[:height, :width]
    nx = (x - width / 2) / max(width / 2, 1)
    ny = (y - height / 2) / max(height / 2, 1)
    shade = np.clip(1.0 - strength * (nx * nx + ny * ny), 0.55, 1.0)
    return np.clip(image.astype(np.float32) * shade[:, :, None], 0, 255).astype(np.uint8)


def draw_comic_burst(image, center, word, box_size):
    """Draw a short comic-book burst and caption centered inside the hand frame."""
    x, y = center
    radius = max(24, int(min(box_size) * 0.22))
    points = []
    for i in range(20):
        angle = 2 * math.pi * i / 20 - math.pi / 2
        r = radius if i % 2 == 0 else int(radius * 0.64)
        points.append([x + int(math.cos(angle) * r), y + int(math.sin(angle) * r)])
    cv2.fillPoly(image, [np.array(points, dtype=np.int32)], (20, 215, 255), cv2.LINE_AA)
    scale = min(1.25, max(0.48, box_size[0] / 300.0))
    thickness = max(1, int(scale * 2))
    (text_width, text_height), baseline = cv2.getTextSize(
        word, cv2.FONT_HERSHEY_COMPLEX, scale, thickness
    )
    origin = (x - text_width // 2, y + text_height // 2)
    cv2.putText(image, word, origin, cv2.FONT_HERSHEY_COMPLEX, scale,
                (20, 20, 20), thickness + 4, cv2.LINE_AA)
    cv2.putText(image, word, origin, cv2.FONT_HERSHEY_COMPLEX, scale,
                (255, 255, 255), thickness, cv2.LINE_AA)


def get_screen_size(fallback_width, fallback_height):
    try:
        user32 = ctypes.windll.user32
        return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    except (AttributeError, OSError):
        return fallback_width, fallback_height


def fit_to_screen(frame, screen_width, screen_height):
    """Scale the camera feed to the display while preserving its aspect ratio."""
    height, width = frame.shape[:2]
    scale = min(screen_width / width, screen_height / height)
    resized_width = max(1, int(width * scale))
    resized_height = max(1, int(height * scale))
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    x = (screen_width - resized_width) // 2
    y = (screen_height - resized_height) // 2
    canvas[y:y + resized_height, x:x + resized_width] = resized
    return canvas


def draw_style_badge(frame, face_box, style_index, enabled):
    """Place a small gallery placard above the detected face."""
    height, width = frame.shape[:2]
    title = ART_STYLES[style_index] if enabled else "GALERIA PAUSADA"
    label = f"SALA {style_index + 1}/4  |  {title}" if enabled else title
    scale = 0.52
    thickness = 1
    (text_width, text_height), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
    )
    badge_width = text_width + 24
    badge_height = text_height + 18

    if face_box is None:
        center_x = width // 2
        baseline = min(height - 8, max(135, height // 3))
    else:
        face_x, face_y, face_width, _face_height = face_box
        center_x = face_x + face_width // 2
        baseline = max(badge_height + 12, face_y - 10)

    x = max(8, min(width - badge_width - 8, center_x - badge_width // 2))
    y = max(8, baseline - badge_height)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + badge_width, y + badge_height),
                  (28, 27, 34), -1)
    cv2.addWeighted(overlay, 0.84, frame, 0.16, 0, frame)
    cv2.rectangle(frame, (x, y), (x + badge_width, y + badge_height),
                  (175, 205, 225), 1, cv2.LINE_AA)
    cv2.putText(frame, label, (x + 12, y + text_height + 8),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (235, 225, 195),
                thickness, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="Filtro visual controlado pelas mãos.")
    parser.add_argument("--camera", type=int, default=0, help="índice da câmera (padrão: 0)")
    parser.add_argument("--no-mic", action="store_true", help="desativa o gatilho de palma")
    parser.add_argument("--threshold", type=float, default=0.32,
                        help="sensibilidade do microfone de 0 a 1 (padrão: 0.32)")
    args = parser.parse_args()

    model = ensure_model()
    camera = open_camera(args.camera)
    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    face_box = None
    face_scan_counter = 0

    clap = ClapDetector(args.threshold)
    if not args.no_mic:
        clap.start()

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model)),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    palette_index = 0
    filters_enabled = True
    style_index = 0
    style_started_at = 0.0
    previous_clap = 0.0
    previous_frame_center = None
    fists_armed_at = None
    spread_active = False
    spread_expanding = False
    spread_strength = 0.0
    palm_return_mode = False
    previous_loop_time = time.monotonic()
    swipe_samples = deque()
    last_swipe_time = 0.0
    manual_override_until = 0.0
    swipe_feedback = ""
    swipe_feedback_until = 0.0
    last_burst_time = 0.0
    burst_word = ""
    burst_until = 0.0
    start_time = time.monotonic()
    last_timestamp = 0
    fullscreen = True
    window_name = "Filtro de Maos"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    first_frame = camera.get(cv2.CAP_PROP_FRAME_WIDTH), camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
    screen_width, screen_height = get_screen_size(
        int(first_frame[0]) or 1280, int(first_frame[1]) or 720
    )

    try:
        with HandLandmarker.create_from_options(options) as landmarker:
            while True:
                try:
                    ok, frame = camera.read()
                except cv2.error as exc:
                    raise SystemExit(
                        "O driver da câmera retornou um quadro inválido. "
                        "Reconecte o DroidCam e tente --camera 0, 1 ou 2. "
                        f"Detalhe: {exc}"
                    ) from exc
                if not ok or frame is None:
                    raise SystemExit(
                        "A câmera parou de enviar vídeo. Confira o DroidCam e tente outro índice."
                    )
                frame = cv2.flip(frame, 1)
                height, width = frame.shape[:2]
                now = time.monotonic()
                face_scan_counter += 1
                if face_scan_counter % 4 == 0 and not face_detector.empty():
                    small_gray = cv2.cvtColor(cv2.resize(frame, None, fx=0.5, fy=0.5),
                                              cv2.COLOR_BGR2GRAY)
                    faces = face_detector.detectMultiScale(
                        small_gray, scaleFactor=1.1, minNeighbors=5,
                        minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE,
                    )
                    if len(faces):
                        fx, fy, fw, fh = max(faces, key=lambda box: box[2] * box[3])
                        face_box = (fx * 2, fy * 2, fw * 2, fh * 2)
                    else:
                        face_box = None

                if clap.last_clap != previous_clap:
                    previous_clap = clap.last_clap
                    palette_index = (palette_index + 1) % len(PALETTES)
                    style_index = (style_index + 1) % len(ART_STYLES)
                    style_started_at = now
                    manual_override_until = now + 12.0

                timestamp = max(last_timestamp + 1, int((now - start_time) * 1000))
                last_timestamp = timestamp
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(image, timestamp)
                frame_dt = min(0.10, max(0.001, now - previous_loop_time))
                previous_loop_time = now
                hand_openness_values = [hand_openness(hand) for hand in result.hand_landmarks]
                palm_centers = [
                    (int(hand[9].x * width), int(hand[9].y * height))
                    for hand in result.hand_landmarks
                ]

                # Two fists arm the gesture; opening both hands triggers the expanding effect.
                both_fists = (len(hand_openness_values) == 2 and
                              all(value <= 0.25 for value in hand_openness_values))
                both_open = (len(hand_openness_values) == 2 and
                             all(value >= 0.75 for value in hand_openness_values))
                if filters_enabled and not spread_active:
                    if both_fists:
                        fists_armed_at = now
                    elif both_open and fists_armed_at is not None and now - fists_armed_at <= 1.8:
                        spread_active = True
                        spread_expanding = True
                        spread_strength = 0.02
                        palm_return_mode = False
                        fists_armed_at = None
                        burst_word = "BOOM!"
                        burst_until = now + 1.2
                        last_burst_time = now
                    elif fists_armed_at is not None and now - fists_armed_at > 1.8:
                        fists_armed_at = None
                if spread_active:
                    if spread_expanding:
                        spread_strength = min(1.0, spread_strength + frame_dt * 2.4)
                        if spread_strength >= 1.0:
                            spread_expanding = False
                    elif len(hand_openness_values) == 2:
                        target_strength = sum(hand_openness_values) / 2.0
                        blend = min(1.0, frame_dt * 3.2)
                        spread_strength += (target_strength - spread_strength) * blend
                        if target_strength <= 0.05 and spread_strength <= 0.10:
                            spread_active = False
                            spread_strength = 0.0
                            palm_return_mode = True

                # A quick horizontal swipe with one open palm selects the next/previous style.
                if (len(result.hand_landmarks) == 1 and
                        hand_openness(result.hand_landmarks[0]) >= 0.75):
                    swipe_x = result.hand_landmarks[0][0].x
                    swipe_samples.append((now, swipe_x))
                    while swipe_samples and now - swipe_samples[0][0] > 0.45:
                        swipe_samples.popleft()
                    if len(swipe_samples) >= 3 and now - last_swipe_time > 1.0:
                        elapsed = now - swipe_samples[0][0]
                        delta_x = swipe_x - swipe_samples[0][1]
                        if 0.12 <= elapsed <= 0.45 and abs(delta_x) >= 0.20:
                            step = 1 if delta_x > 0 else -1
                            style_index = (style_index + step) % len(ART_STYLES)
                            style_started_at = now
                            manual_override_until = now + 12.0
                            last_swipe_time = now
                            swipe_feedback = "Tapa para a direita: proximo estilo" if step > 0 else "Tapa para a esquerda: estilo anterior"
                            swipe_feedback_until = now + 1.4
                            swipe_samples.clear()
                else:
                    swipe_samples.clear()

                accent = PALETTES[palette_index]
                display_frame = frame.copy()
                polygon = find_hand_frame(result.hand_landmarks, width, height)
                if polygon is not None:
                    palm_return_mode = False
                if filters_enabled and polygon is not None and style_started_at == 0.0:
                    style_started_at = now
                if (filters_enabled and polygon is not None and not spread_active and
                        now >= manual_override_until and now - style_started_at >= 6.0):
                    style_index = (style_index + 1) % len(ART_STYLES)
                    style_started_at = now

                mask = np.zeros((height, width), dtype=np.uint8)
                if polygon is not None:
                    cv2.fillConvexPoly(mask, polygon, 255)
                palm_radius = max(42, int(min(width, height) * 0.09))
                if spread_active:
                    radius = palm_radius + int(max(width, height) * spread_strength)
                    for center_hand in palm_centers:
                        cv2.circle(mask, center_hand, radius, 255, -1, cv2.LINE_AA)
                elif palm_return_mode:
                    for center_hand in palm_centers:
                        cv2.circle(mask, center_hand, palm_radius, 255, -1, cv2.LINE_AA)

                if filters_enabled and np.any(mask):
                    art_frame = apply_art_filter(frame, style_index)
                    display_frame[mask > 0] = art_frame[mask > 0]
                    if polygon is not None:
                        cv2.polylines(display_frame, [polygon], True, accent, 3, cv2.LINE_AA)
                        box = cv2.boundingRect(polygon)
                        plaque = display_frame.copy()
                        plaque_y = min(height - 8, box[1] + box[3] - 14)
                        cv2.rectangle(plaque, (box[0] + 8, plaque_y - 22),
                                      (box[0] + min(box[2] - 8, 245), plaque_y + 5),
                                      (24, 25, 32), -1)
                        label = f"SALA {style_index + 1}/4 | {ART_STYLES[style_index]}"
                        cv2.putText(plaque, label, (box[0] + 15, plaque_y - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (225, 220, 195),
                                    1, cv2.LINE_AA)
                        display_frame[mask > 0] = plaque[mask > 0]
                        center = tuple(np.mean(polygon, axis=0).astype(int))
                        moved = (previous_frame_center is not None and
                                 math.hypot(center[0] - previous_frame_center[0],
                                            center[1] - previous_frame_center[1]) > 22)
                        if (previous_frame_center is None or moved) and now - last_burst_time > 0.8:
                            burst_word = random.choice(("POW!", "BAM!", "ZAP!", "THWIP!", "BOOM!"))
                            burst_until = now + 0.85
                            last_burst_time = now
                        previous_frame_center = center
                    else:
                        previous_frame_center = None

                    show_burst = now < burst_until and (not spread_active or spread_strength >= 0.30)
                    if show_burst:
                        burst_layer = display_frame.copy()
                        if spread_active and palm_centers:
                            burst_center = tuple(np.mean(palm_centers, axis=0).astype(int))
                            burst_size = (width, height)
                        elif polygon is not None:
                            burst_center = tuple(np.mean(polygon, axis=0).astype(int))
                            box = cv2.boundingRect(polygon)
                            burst_size = (box[2], box[3])
                        else:
                            burst_center = (width // 2, height // 2)
                            burst_size = (width, height)
                        draw_comic_burst(burst_layer, burst_center, burst_word, burst_size)
                        display_frame[mask > 0] = burst_layer[mask > 0]

                if spread_active and spread_expanding:
                    status = "EXPLOSAO: abrindo o filtro pela camera toda"
                elif spread_active:
                    status = "Feche as duas maos devagar para recolher o filtro"
                elif palm_return_mode and palm_centers:
                    status = "Filtro voltou para as palmas; faca dois Ls para enquadrar"
                elif polygon is not None:
                    status = "Moldura detectada: filtro aplicado dentro das maos"
                elif filters_enabled:
                    status = "Punhos fechados -> abra as duas maos para expandir"
                else:
                    status = "Galeria desligada: pressione F para ativar"

                if now < swipe_feedback_until:
                    status = swipe_feedback
                cv2.rectangle(display_frame, (12, 12), (width - 12, 108), (18, 18, 24), -1)
                cv2.putText(display_frame, "FILTRO DE MAOS", (26, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.72, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(display_frame, status, (26, 66), cv2.FONT_HERSHEY_SIMPLEX,
                            0.48, accent, 1, cv2.LINE_AA)
                mode_label = "GALERIA: ON" if filters_enabled else "GALERIA: OFF"
                cv2.putText(display_frame, f"{mode_label}  |  F: galeria  M: tela  Q: sair", (width - 330, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
                cv2.putText(display_frame,
                            "MAO ABERTA: TAPA DIREITA = PROXIMO | TAPA ESQUERDA = ANTERIOR",
                            (26, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.39,
                            (225, 220, 195), 1, cv2.LINE_AA)
                draw_style_badge(display_frame, face_box, style_index, filters_enabled)
                output_frame = fit_to_screen(display_frame, screen_width, screen_height)
                cv2.imshow(window_name, output_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("f"):
                    filters_enabled = not filters_enabled
                if key == ord("m"):
                    fullscreen = not fullscreen
                    if fullscreen:
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                                              cv2.WINDOW_FULLSCREEN)
                    else:
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                                              cv2.WINDOW_NORMAL)
                        cv2.resizeWindow(window_name, 960, 540)
    finally:
        clap.stop()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

