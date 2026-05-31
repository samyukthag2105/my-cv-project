import cv2
import json
import csv
import time
import math
import mediapipe as mp
import pyttsx3

# Text-to-speech
engine = pyttsx3.init()

def say(text):
    engine.say(text)
    engine.runAndWait()


# Face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("model.yml")

# Load labels
with open("labels.json") as f:
    label_names = json.load(f)

# Face detector
cascade_path = (
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)
detector = cv2.CascadeClassifier(cascade_path)

THRESHOLD = 70

# Face Mesh
face_mesh = mp.solutions.face_mesh.FaceMesh(
    refine_landmarks=True,
    max_num_faces=1
)

# Eye landmarks
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

EAR_THRESHOLD = 0.21


def ear(lm, idx, w, h):
    p = [(lm[i].x * w, lm[i].y * h) for i in idx]

    d = lambda a, b: math.hypot(
        a[0] - b[0],
        a[1] - b[1]
    )

    return (
        d(p[1], p[5]) +
        d(p[2], p[4])
    ) / (2 * d(p[0], p[3]))


def log_attendance(name):
    with open("attendance.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            name,
            time.strftime("%Y-%m-%d %H:%M:%S")
        ])


# Camera
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

state = "LOOKING"      # LOOKING -> CHALLENGE -> DONE
pending = None
closed = False
last_log = 0

say("Please look at the camera")

while True:

    ok, frame = camera.read()

    if not ok:
        break

    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Step 1: Face Recognition
    if state == "LOOKING":

        for (x, y, fw, fh) in detector.detectMultiScale(
            gray,
            1.1,
            5
        ):

            face = cv2.resize(
                gray[y:y + fh, x:x + fw],
                (200, 200)
            )

            label, conf = recognizer.predict(face)

            if conf < THRESHOLD:
                pending = label_names[str(label)]
                state = "CHALLENGE"

                say(
                    pending +
                    ", please blink to confirm"
                )

            cv2.rectangle(
                frame,
                (x, y),
                (x + fw, y + fh),
                (0, 255, 0),
                2
            )

    # Step 2: Blink Detection
    elif state == "CHALLENGE":

        cv2.putText(
            frame,
            "Please blink...",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        result = face_mesh.process(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )

        if result.multi_face_landmarks:

            lm = result.multi_face_landmarks[0].landmark

            avg_ear = (
                ear(lm, LEFT_EYE, w, h) +
                ear(lm, RIGHT_EYE, w, h)
            ) / 2

            if avg_ear < EAR_THRESHOLD and not closed:
                closed = True

            elif avg_ear >= EAR_THRESHOLD and closed:

                closed = False

                if time.time() - last_log > 5:

                    log_attendance(pending)
                    last_log = time.time()

                    say(
                        f"Thank you {pending}, "
                        "you are checked in"
                    )

                    state = "DONE"

    # Step 3: Attendance Marked
    elif state == "DONE":

        cv2.putText(
            frame,
            "Checked in! R = Reset, Q = Quit",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    cv2.imshow("Site Check-In", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key == ord("r"):
        state = "LOOKING"
        pending = None
        closed = False

camera.release()
cv2.destroyAllWindows()