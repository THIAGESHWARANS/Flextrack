import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)

count = 0
stage = None  # "open" or "close"

def get_angle(a, b, c):
    """Helper to calculate angle between three points."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360 - angle
    return angle

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame for natural display
    frame = cv2.flip(frame, 1)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)

    image.flags.writeable = True
    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Get key landmark coordinates
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        # Calculate horizontal distance between ankles and wrists
        ankle_distance = abs(left_ankle.x - right_ankle.x)
        wrist_height = (left_wrist.y + right_wrist.y) / 2
        shoulder_height = (left_shoulder.y + right_shoulder.y) / 2

        # Jumping Jack detection logic
        is_open = (
            ankle_distance > 0.5 and  # legs apart
            wrist_height < shoulder_height  # hands above shoulders
        )
        is_closed = (
            ankle_distance < 0.3 and  # legs together
            wrist_height > shoulder_height  # hands down
        )

        if is_open and stage == "closed":
            stage = "open"
        elif is_closed and stage == "open":
            stage = "closed"
            count += 1

        # Draw landmarks and show count
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.putText(frame, f"Jumping Jacks Count: {count}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    cv2.imshow("Jumping Jacks Tracker", frame)
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
