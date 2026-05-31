import cv2
import os

# Get user's name
name = input("Enter your name: ").strip()

# Create a folder for the user's images
save_dir = os.path.join("dataset", name)
os.makedirs(save_dir, exist_ok=True)

# Load the face detector
cascade_path = (
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

face_detector = cv2.CascadeClassifier(cascade_path)

# Start the webcam
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

count = 0

while count < 30:
    # Read a frame from the camera
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
        count += 1

        # Extract and resize the face region
        face_img = cv2.resize(
            gray[y:y + h, x:x + w],
            (200, 200)
        )

        # Save the image
        cv2.imwrite(
            os.path.join(save_dir, f"{count}.png"),
            face_img
        )

        # Draw a rectangle around the face
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    # Show the video feed
    cv2.imshow(
        "Enrolling - Move Slightly. Press Q to Stop",
        frame
    )

    # Exit if Q is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Clean up
camera.release()
cv2.destroyAllWindows()

print(f"Saved {count} images to {save_dir}")