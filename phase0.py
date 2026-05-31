import cv2

# OpenCV ships with a basic pre-trained face detector.
cascade_path = (
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

face_detector = cv2.CascadeClassifier(cascade_path)

# CAP_DSHOW makes the camera start reliably on Windows.
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    # Read one image from the camera
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

    # Draw a rectangle around each detected face
    for (x, y, w, h) in faces:
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    # Display the result
    cv2.imshow("Phase 0 - Press Q to Quit", frame)

    # Exit when Q is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Clean up
camera.release()
cv2.destroyAllWindows()