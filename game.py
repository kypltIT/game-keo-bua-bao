import random
import cv2
import numpy as np
import hand_detection_lib as handlib
import os

# ==== Config & assets ====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR = os.path.join(BASE_DIR, "pix")  # đảm bảo tuyệt đối

def load_icon(kind: int):
    path = os.path.join(ASSET_DIR, f"{kind}.png")
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # giữ alpha nếu có
    if img is None:
        raise FileNotFoundError(f"Không đọc được icon: {path} (kiểm tra tên file 0.png/1.png/2.png)")
    return img

ICONS = {k: load_icon(k) for k in (0, 1, 2)}  # 0: Lá, 1: Đấm, 2: Kéo
LABELS = {0: "Paper", 1: "Rock", 2: "Scissors"}

detector = handlib.handDetector()
cam = cv2.VideoCapture(0)

def overlay_png(dst, src, x, y):
    """Dán src (có thể 4 kênh) lên dst tại (x,y), có cắt mép và trộn alpha."""
    h, w = src.shape[:2]
    H, W = dst.shape[:2]
    if x >= W or y >= H:
        return
    w = min(w, W - x)
    h = min(h, H - y)
    src = src[:h, :w]
    roi = dst[y:y+h, x:x+w]

    if src.shape[2] == 4:
        alpha = (src[:, :, 3] / 255.0)[:, :, None]
        fg = src[:, :, :3].astype(np.float32)
        bg = roi.astype(np.float32)
        blended = alpha * fg + (1 - alpha) * bg
        dst[y:y+h, x:x+w] = blended.astype(np.uint8)
    else:
        dst[y:y+h, x:x+w] = src

def draw_results(frame, user_draw):
    # Guard: chỉ chấp nhận 0/1/2
    if user_draw not in (0, 1, 2):
        cv2.putText(frame, "Khong nhan dien cu chi hop le", (50, 550),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        return frame

    com_draw = random.randint(0, 2)

    # You
    cv2.putText(frame, 'You', (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2, cv2.LINE_AA)
    try:
        s_img = ICONS[user_draw]
        overlay_png(frame, s_img, 50, 100)
        cv2.putText(frame, LABELS[user_draw], (50, 100 + s_img.shape[0] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    except Exception as e:
        cv2.putText(frame, f"Loi icon user: {e}", (50, 520),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Computer
    cv2.putText(frame, 'Computer', (400, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2, cv2.LINE_AA)
    try:
        s_img = ICONS[com_draw]
        overlay_png(frame, s_img, 400, 100)
        cv2.putText(frame, LABELS[com_draw], (400, 100 + s_img.shape[0] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    except Exception as e:
        cv2.putText(frame, f"Loi icon com: {e}", (400, 520),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Kết quả
    if user_draw == com_draw:
        result = "DRAW!"
    elif (user_draw == 0 and com_draw == 1) or \
         (user_draw == 1 and com_draw == 2) or \
         (user_draw == 2 and com_draw == 0):
        result = "YOU WIN!"
    else:
        result = "YOU LOSE!"
    cv2.putText(frame, result, (50, 550), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 0, 255), 2, cv2.LINE_AA)
    return frame

while True:
    ret, frame = cam.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    frame, hand_lms = detector.findHands(frame)
    n_fingers = detector.count_finger(hand_lms)

    # 0: Lá (Paper), 1: Đấm (Rock), 2: Kéo (Scissors)
    user_draw = -1
    if n_fingers == 0:
        user_draw = 1
    elif n_fingers == 2:
        user_draw = 2
    elif n_fingers == 5:
        user_draw = 0
    elif n_fingers != -1:
        cv2.putText(frame, "Chi chap nhan Dam/Lá/Keo", (50, 520),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    else:
        cv2.putText(frame, "Khong thay ban tay", (50, 520),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("game", frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    elif key == ord(" "):
        # Chỉ hiển thị kết quả khi cử chỉ hợp lệ
        frame = draw_results(frame, user_draw)
        cv2.imshow("game", frame)
        cv2.waitKey(0)  # dừng đến khi nhấn phím bất kỳ

cam.release()
cv2.destroyAllWindows()
