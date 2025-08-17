import cv2
import mediapipe as mp

class handDetector():
    def __init__(self):
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=False,
            max_num_hands=2,  # Allow detection of both hands
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img):
        # Convert from BGR to RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process with mediapipe
        results = self.hands.process(imgRGB)
        hand_lms = []
        handedness = None

        if results.multi_hand_landmarks and results.multi_handedness:
            # Draw landmarks for all detected hands
            for handlm in results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img, handlm, self.mpHands.HAND_CONNECTIONS)

            # Use the first detected hand (most confident)
            firstHand = results.multi_hand_landmarks[0]
            handedness = results.multi_handedness[0].classification[0].label  # "Left" or "Right"
            
            h, w, _ = img.shape
            for id, lm in enumerate(firstHand.landmark):
                real_x, real_y = int(lm.x * w), int(lm.y * h)
                hand_lms.append([id, real_x, real_y])

        return img, hand_lms, handedness

    def count_finger(self, hand_lms, handedness=None):
        finger_start_index = [4, 8, 12, 16, 20]
        n_fingers = 0

        if len(hand_lms) > 0:
            # Check thumb - logic differs for left vs right hand
            if handedness == "Left":
                # For left hand, thumb is up when x coordinate is greater
                if hand_lms[finger_start_index[0]][1] > hand_lms[finger_start_index[0]-1][1]:
                    n_fingers += 1
            else:
                # For right hand (default), thumb is up when x coordinate is smaller
                if hand_lms[finger_start_index[0]][1] < hand_lms[finger_start_index[0]-1][1]:
                    n_fingers += 1

            # Check the other 4 fingers (same logic for both hands)
            for idx in range(1, 5):
                if hand_lms[finger_start_index[idx]][2] < hand_lms[finger_start_index[idx]-2][2]:
                    n_fingers += 1

            return n_fingers
        else:
            return -1




