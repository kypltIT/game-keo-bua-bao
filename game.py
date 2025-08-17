import random
import cv2
import numpy as np
import hand_detection_lib as handlib
import os
import time

# ==== Config & assets ====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR = os.path.join(BASE_DIR, "pix")

def load_icon(kind: int):
    path = os.path.join(ASSET_DIR, f"{kind}.png")
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot load icon: {path} (check file names 0.png/1.png/2.png)")
    return img

ICONS = {k: load_icon(k) for k in (0, 1, 2)}  # 0: Paper, 1: Rock, 2: Scissors
LABELS = {0: "Paper", 1: "Rock", 2: "Scissors"}

detector = handlib.handDetector()
cam = cv2.VideoCapture(0)

# Game state variables
game_state = "playing"  # "playing", "computer_thinking", "showing_result"
result_start_time = 0
computer_thinking_start_time = 0
last_gesture = -1
gesture_stable_time = 0
final_computer_choice = 0
GESTURE_HOLD_TIME = 1.0  # seconds to hold gesture before triggering result
COMPUTER_THINKING_TIME = 0.5  # seconds for computer animation

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

def draw_computer_thinking(frame, user_draw):
    """Draw computer thinking animation with random choices"""
    # Show user choice
    cv2.putText(frame, 'You', (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2, cv2.LINE_AA)
    try:
        s_img = ICONS[user_draw]
        overlay_png(frame, s_img, 50, 100)
        cv2.putText(frame, LABELS[user_draw], (50, 100 + s_img.shape[0] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    except Exception as e:
        cv2.putText(frame, f"User icon error: {e}", (50, 520),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Show computer thinking with random animation
    cv2.putText(frame, 'Computer', (400, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2, cv2.LINE_AA)
    
    # Random choice for animation
    random_choice = random.randint(0, 2)
    try:
        s_img = ICONS[random_choice]
        overlay_png(frame, s_img, 400, 100)
        cv2.putText(frame, LABELS[random_choice], (400, 100 + s_img.shape[0] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    except Exception as e:
        cv2.putText(frame, f"Computer icon error: {e}", (400, 520),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Thinking indicator
    cv2.putText(frame, "Computer is thinking...", (200, 400), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 0), 2, cv2.LINE_AA)
    
    return frame

def draw_results(frame, user_draw, com_draw):
    """Draw final results"""
    # You
    cv2.putText(frame, 'You', (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2, cv2.LINE_AA)
    try:
        s_img = ICONS[user_draw]
        overlay_png(frame, s_img, 50, 100)
        cv2.putText(frame, LABELS[user_draw], (50, 100 + s_img.shape[0] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    except Exception as e:
        cv2.putText(frame, f"User icon error: {e}", (50, 520),
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
        cv2.putText(frame, f"Computer icon error: {e}", (400, 520),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Result
    if user_draw == com_draw:
        result = "DRAW!"
        color = (0, 255, 255)
    elif (user_draw == 0 and com_draw == 1) or \
         (user_draw == 1 and com_draw == 2) or \
         (user_draw == 2 and com_draw == 0):
        result = "YOU WIN!"
        color = (0, 255, 0)
    else:
        result = "YOU LOSE!"
        color = (0, 0, 255)
    
    cv2.putText(frame, result, (200, 400), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, color, 3, cv2.LINE_AA)
    
    # Play again instruction
    cv2.putText(frame, "Press 'R' to Play Again", (180, 450), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2, cv2.LINE_AA)
    
    return frame

def draw_instructions(frame, handedness=None):
    
    # Show hand status more subtly
    if handedness:
        cv2.putText(frame, f"{handedness} Hand", (50, 510), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1, cv2.LINE_AA)

while True:
    ret, frame = cam.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    frame, hand_lms, handedness = detector.findHands(frame)
    n_fingers = detector.count_finger(hand_lms, handedness)

    # 0: Paper, 1: Rock, 2: Scissors
    user_draw = -1
    if n_fingers == 0:
        user_draw = 1  # Rock
    elif n_fingers == 2:
        user_draw = 2  # Scissors
    elif n_fingers == 5:
        user_draw = 0  # Paper

    current_time = time.time()
    
    if game_state == "playing":
        draw_instructions(frame, handedness)
        
        # Auto-trigger result when gesture is held stable
        if user_draw in (0, 1, 2):
            if user_draw == last_gesture:
                if current_time - gesture_stable_time >= GESTURE_HOLD_TIME:
                    game_state = "computer_thinking"
                    computer_thinking_start_time = current_time
                    final_computer_choice = random.randint(0, 2)  # Set final choice
            else:
                last_gesture = user_draw
                gesture_stable_time = current_time
                
            # Show countdown
            remaining_time = GESTURE_HOLD_TIME - (current_time - gesture_stable_time)
            if remaining_time > 0 and user_draw == last_gesture:
                cv2.putText(frame, f"Hold: {remaining_time:.1f}s", (50, 320),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            last_gesture = -1
            
    elif game_state == "computer_thinking":
        frame = draw_computer_thinking(frame, last_gesture)
        
        # Move to result after computer thinking time
        if current_time - computer_thinking_start_time >= COMPUTER_THINKING_TIME:
            game_state = "showing_result"
            result_start_time = current_time
            
    elif game_state == "showing_result":
        frame = draw_results(frame, last_gesture, final_computer_choice)
        
        # Auto return to playing after 4 seconds
        if current_time - result_start_time >= 4.0:
            game_state = "playing"
            last_gesture = -1

    cv2.imshow("Rock Paper Scissors Game", frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == 27:  # ESC key
        break
    elif key == ord("r") or key == ord("R"):
        # Restart game
        game_state = "playing"
        last_gesture = -1
        cv2.putText(frame, "Game Restarted!", (250, 300),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Rock Paper Scissors Game", frame)
        cv2.waitKey(500)  # Show restart message briefly

cam.release()
cv2.destroyAllWindows()
