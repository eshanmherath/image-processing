import cv2

cap = cv2.VideoCapture(0)  # Get default video input
cap.set(3, 640)  # Set Width
cap.set(4, 480)  # Set Height

# Replace below path with path to haarcascade_frontalface_default.xml file in your opencv installed directory
face_cascade = cv2.CascadeClassifier(
    'C:\\Users\Eshan\Anaconda3\pkgs\opencv-3.3.0-py36_200\Library\etc\haarcascades\haarcascade_frontalface_default.xml')

while True:
    # Read each frame
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw a rectangle around the detected face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)

    cv2.imshow('frame', frame)

    # Stop Condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
