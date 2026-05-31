import cv2
import json

# Load the trained face recognition model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("model.yml")

# Load label-to-name mappings
with open("labels.json", "r") as f:
    label_names = json.load(f)

# Load the face detector
cascade_path = (
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

face_detector = cv2.CascadeClassifier(cascade_path)

# Start the webcam
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Lower threshold = stricter recognition
THRESHOLD = 180

while True:
    # Capture a frame
    ok, frame = camera.read()

    if not ok:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:
        # Extract and resize face region
        face_img = cv2.resize(
            gray[y:y + h, x:x + w],
            (200, 200)
        )

        # Predict identity
        label, confidence = recognizer.predict(face_img)

        if confidence < THRESHOLD:
            text = f"{label_names[str(label)]} ({confidence:.0f})"
            color = (0, 255, 0)  # Green
        else:
            text = f"Unknown ({confidence:.0f})"
            color = (0, 0, 255)  # Red

        # Draw bounding box
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            color,
            2
        )

        # Display name and confidence
        cv2.putText(
            frame,
            text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    # Show the video feed
    cv2.imshow(
        "Recognize - Press Q to Quit",
        frame
    )

    # Exit when Q is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Clean up
camera.release()
cv2.destroyAllWindows()