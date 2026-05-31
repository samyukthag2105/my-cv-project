import cv2
import math
import mediapipe as mp

# Initialize MediaPipe Face Mesh
face_mesh = mp.solutions.face_mesh.FaceMesh(
    refine_landmarks=True,
    max_num_faces=1
)

# Six landmark points around each eye
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def ear(landmarks, indices, width, height):
    """
    Calculate Eye Aspect Ratio (EAR)
    """
    points = [
        (landmarks[i].x * width, landmarks[i].y * height)
        for i in indices
    ]

    def distance(a, b):
        return math.hypot(
            a[0] - b[0],
            a[1] - b[1]
        )

    return (
        distance(points[1], points[5]) +
        distance(points[2], points[4])
    ) / (2 * distance(points[0], points[3]))


# Start webcam
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Tune this value for your eyes/camera
EAR_THRESHOLD = 0.21

blinks = 0
closed = False

while True:
    ok, frame = camera.read()

    if not ok:
        break

    height, width = frame.shape[:2]

    # Convert BGR to RGB for MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process face landmarks
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0].landmark

        # Average EAR from both eyes
        avg_ear = (
            ear(landmarks, LEFT_EYE, width, height) +
            ear(landmarks, RIGHT_EYE, width, height)
        ) / 2

        # Blink detection logic
        if avg_ear < EAR_THRESHOLD and not closed:
            closed = True

        elif avg_ear >= EAR_THRESHOLD and closed:
            blinks += 1
            closed = False

        # Display EAR and blink count
        cv2.putText(
            frame,
            f"EAR: {avg_ear:.2f}  Blinks: {blinks}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    cv2.imshow(
        "Blink Test - Press Q to Quit",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Clean up
camera.release()
cv2.destroyAllWindows()