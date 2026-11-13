"""摄像头互动小游戏：用嘴巴接住兔耳，绿框 +1、红框 -1，按 q 退出。"""

import os
import random
import sys
import time

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WINDOW = "Face Game"
FRAME_W, FRAME_H = 1280, 720
DROP_W, DROP_H = 60, 40
FONT = cv2.FONT_HERSHEY_SIMPLEX

SPAWN_INTERVAL = 1.5
MAX_ITEMS = 5
SPEED_START, SPEED_STEP, SPEED_MIN, SPEED_MAX = 2.0, 0.2, 1.0, 8.0
COLLISION_RATIO = 0.5
MOUTH_H, MOUTH_W = 0.3, 0.6
SMOOTH = 0.7
MILESTONE_AT = (5, -5)
TOAST_SECONDS = 1.5
END_SCORE = 10
SHOW_DEBUG = True

GREEN, RED, BLUE, WHITE = (0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 255)

CN_FONTS = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "/System/Library/Fonts/PingFang.ttc",
)
CN_FONT = next((p for p in CN_FONTS if os.path.exists(p)), None)


def put_text(img, label, pos, size, color, center=False):
    """中文优先，缺字体时回退到英文；label 为 (中文, 英文)。"""
    cn, en = label
    use_cn = CN_FONT is not None and Image is not None
    text = cn if use_cn else en
    if center:
        if use_cn:
            w, h = ImageFont.truetype(CN_FONT, size).getbbox(text)[2:]
        else:
            (w, h), _ = cv2.getTextSize(text, FONT, size / 30, 2)
        pos = (int(pos[0] - w / 2), int(pos[1] - h / 2))
    if use_cn:
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ImageDraw.Draw(pil).text(pos, text, font=ImageFont.truetype(CN_FONT, size),
                                 fill=color[::-1])
        img[:] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    else:
        cv2.putText(img, text, pos, FONT, size / 30, color, 2, cv2.LINE_AA)


def load_detector():
    """加载 Haar 人脸检测器。"""
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    det = cv2.CascadeClassifier(path)
    if det.empty():
        raise RuntimeError("人脸检测器加载失败，请安装 opencv-python")
    return det


def load_sprites():
    """读取干净版与带背景版兔耳，缺文件时从原图生成。"""
    def read(name):
        img = cv2.imread(os.path.join(BASE_DIR, name))
        if img is None:
            raise FileNotFoundError(f"缺少图片: {name}")
        return cv2.resize(img, (DROP_W, DROP_H))

    if all(os.path.exists(os.path.join(BASE_DIR, n))
           for n in ("rabbit_ears_clean.png", "rabbit_ears_dirty.png")):
        return read("rabbit_ears_clean.png"), read("rabbit_ears_dirty.png")

    src = read("rabbit_ears.png")
    fg = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY) > 30
    clean, dirty = src.copy(), src.copy()
    clean[~fg], dirty[~fg] = WHITE, (0, 0, 0)
    return clean, dirty


def make_alpha(dirty):
    """借黑底版亮度取前景，避免干净版白底与耳内高光混淆。"""
    fg = (cv2.cvtColor(dirty, cv2.COLOR_BGR2GRAY) > 40).astype(np.float32)
    return cv2.GaussianBlur(fg, (3, 3), 0)


def blit(frame, x, y, sprite, alpha):
    """贴图并裁剪到画面内，float 坐标与负坐标都安全。"""
    h, w = sprite.shape[:2]
    x, y = int(x), int(y)
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, frame.shape[1]), min(y + h, frame.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    sx, sy = x0 - x, y0 - y
    src = sprite[sy:sy + y1 - y0, sx:sx + x1 - x0]
    dst = frame[y0:y1, x0:x1]
    if alpha is None:
        dst[:] = src
    else:
        a = alpha[sy:sy + y1 - y0, sx:sx + x1 - x0][..., None]
        dst[:] = (src * a + dst * (1 - a)).astype(np.uint8)


def mouth_rect(face, w, h):
    """人脸下半部分居中区域，裁剪到画面内。"""
    x, y, fw, fh = face
    return (max(int(x + fw * (0.5 - MOUTH_W / 2)), 0),
            max(int(y + fh * (1 - MOUTH_H)), 0),
            min(int(x + fw * (0.5 + MOUTH_W / 2)), w),
            min(int(y + fh), h))


def overlap_ratio(item_box, mouth):
    """重叠面积占道具面积的比例。"""
    x1, y1 = max(item_box[0], mouth[0]), max(item_box[1], mouth[1])
    x2, y2 = min(item_box[2], mouth[2]), min(item_box[3], mouth[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    area = (item_box[2] - item_box[0]) * (item_box[3] - item_box[1])
    return (x2 - x1) * (y2 - y1) / area if area > 0 else 0.0


class FaceGame:
    """游戏状态与帧循环。"""

    def __init__(self, cap, detector, clean, dirty):
        self.cap, self.detector = cap, detector
        self.sprites = {True: (clean, make_alpha(dirty)), False: (dirty, None)}
        self.score = 0
        self.speed = SPEED_START
        self.items = []
        self.face = None
        self.milestones = set()
        self.toast = None
        self.last_spawn = time.time()
        self.frame = None
        self.w, self.h = FRAME_W, FRAME_H

    def _update_face(self, display):
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, 1.1, 6, minSize=(80, 80))
        if len(faces) == 0:
            self.face = None
            return
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        cur = (float(x), float(y), float(w), float(h))
        self.face = (cur if self.face is None else
                     tuple(SMOOTH * a + (1 - SMOOTH) * b for a, b in zip(cur, self.face)))
        if SHOW_DEBUG:
            fx, fy, fw, fh = [int(v) for v in self.face]
            cv2.rectangle(display, (fx, fy), (fx + fw, fy + fh), BLUE, 2)
            m = mouth_rect(self.face, self.w, self.h)
            cv2.rectangle(display, (m[0], m[1]), (m[2], m[3]), GREEN, 2)

    def _spawn(self, now):
        x = random.randint(20, max(21, self.w - DROP_W - 20))
        self.items.append([float(x), float(-DROP_H), random.choice((True, False))])
        self.last_spawn = now

    def _update_items(self, dt, mouth):
        # 按时间位移，保证不同帧率下速度一致
        alive = []
        for item in self.items:
            item[1] += self.speed * 60 * dt
            if item[1] > self.h:
                continue
            x, y = int(item[0]), int(item[1])
            box = (x, y, x + DROP_W, y + DROP_H)
            if mouth is not None and overlap_ratio(box, mouth) >= COLLISION_RATIO:
                good = item[2]
                self.score += 1 if good else -1
                delta = SPEED_STEP if good else -SPEED_STEP
                self.speed = float(np.clip(self.speed + delta, SPEED_MIN, SPEED_MAX))
                continue
            alive.append(item)
        self.items = alive

    def _draw_items(self, display):
        for x, y, good in self.items:
            sprite, alpha = self.sprites[good]
            blit(display, x, y, sprite, alpha)
            xi, yi = int(x), int(y)
            cv2.rectangle(display, (xi, yi), (xi + DROP_W, yi + DROP_H),
                          GREEN if good else RED, 2)

    def _set_toast(self, label):
        """预渲染飘字图层并记录到期时间，不阻塞主循环。"""
        overlay = np.zeros_like(self.frame)
        size = max(36, min(90, int(self.w * 0.09)))
        put_text(overlay, label, (self.w // 2, self.h // 4), size, GREEN, center=True)
        mask = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY) > 0
        self.toast = (overlay, mask, time.time() + TOAST_SECONDS)

    def _draw_toast(self, display):
        """未过期时把飘字贴到画面上。"""
        if self.toast is None:
            return
        overlay, mask, expire = self.toast
        if time.time() >= expire:
            self.toast = None
            return
        display[mask] = overlay[mask]

    def _finish(self, display):
        win = self.score > 0
        size = max(36, min(90, int(self.w * 0.09)))
        base = cv2.addWeighted(display, 0.35, np.zeros_like(display), 0.65, 0)
        put_text(base, ("你赢了！", "YOU WIN!") if win else ("游戏结束", "GAME OVER"),
                 (self.w // 2, self.h // 2 - 20), size,
                 GREEN if win else RED, center=True)
        put_text(base, (f"最终得分 {self.score}", f"Score {self.score}"),
                 (self.w // 2, self.h // 2 + 70), max(24, size // 3), WHITE, center=True)
        cv2.imshow(WINDOW, base)
        cv2.waitKey(3000)

    def run(self):
        """主循环。"""
        prev = time.time()
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            self.frame = cv2.flip(frame, 1)
            self.h, self.w = self.frame.shape[:2]
            now = time.time()
            dt = min(now - prev, 0.05)
            prev = now

            display = self.frame.copy()
            self._update_face(display)
            if now - self.last_spawn >= SPAWN_INTERVAL and len(self.items) < MAX_ITEMS:
                self._spawn(now)

            mouth = mouth_rect(self.face, self.w, self.h) if self.face else None
            self._update_items(dt, mouth)

            if self.score in MILESTONE_AT and self.score not in self.milestones:
                self.milestones.add(self.score)
                self._set_toast(("666", "NOOB") if self.score > 0 else ("没实力", "666"))

            self._draw_items(display)
            self._draw_toast(display)
            cv2.putText(display, f"Score: {self.score}", (30, 40),
                        FONT, 1, RED, 2, cv2.LINE_AA)
            cv2.putText(display, "GREEN +1 | RED -1 | Q quit", (30, self.h - 20),
                        FONT, 0.6, WHITE, 1, cv2.LINE_AA)

            if abs(self.score) >= END_SCORE:
                self._finish(display)
                break
            cv2.imshow(WINDOW, display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


def main():
    """入口：加载资源并启动，异常时保证释放摄像头。"""
    try:
        detector = load_detector()
        clean, dirty = load_sprites()
    except (RuntimeError, FileNotFoundError) as e:
        print(f"初始化失败: {e}")
        return 1

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    print("绿框 +1，红框 -1，按 q 退出")
    try:
        FaceGame(cap, detector, clean, dirty).run()
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
