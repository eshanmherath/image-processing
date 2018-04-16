import time

import cv2

cap = cv2.VideoCapture(0)  # Get default video input
cap.set(3, 640)  # Set Width
cap.set(4, 480)  # Set Height

# Replace below path with path to haarcascade_frontalface_default.xml file in your opencv installed directory
face_cascade = cv2.CascadeClassifier(
    'C:\\Users\Eshan\Anaconda3\pkgs\opencv-3.3.0-py36_200\Library\etc\haarcascades\haarcascade_frontalface_default.xml')

period = 5  # seconds
count = 0

# Find OpenCV version
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

# With webcam get(CV_CAP_PROP_FPS) does not work.
# Let's see for ourselves.

last_capture = time.time()

while True:
    # Press q key on keyboard to stop the program
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

    # Read each frame
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw a rectangle around the detected face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)

    cv2.imshow('frame', frame)

    if time.time() - last_capture > period:
        cv2.imwrite('frame%d.jpg' % count, frame)
        last_capture = time.time()
        print('frame%d.jpg saved successfully' % count)
        count += 1

    # Stop Condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
