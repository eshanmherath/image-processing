import os
import cv2
import time

import numpy as np
import tensorflow as tf

from utilities import get_images_formatted

model_path = os.path.dirname(os.path.realpath(__file__)) + '\\models\\'
image_dir_path = os.path.dirname(os.path.realpath(__file__)) + '\\real_time_images\\'

# Replace below path with path to haarcascade_frontalface_default.xml file in your opencv installed directory
face_cascade = cv2.CascadeClassifier(
    'C:\\Users\Eshan\Anaconda3\pkgs\opencv-3.3.0-py36_200\Library\etc\haarcascades\haarcascade_frontalface_default.xml')

saver = None
sess = tf.Session()

try:
    saver = tf.train.import_meta_graph(model_path + 'dogs-cats-model.meta')
except IOError as e:
    print('No Trained Model Found at %s. \nPlease run train.py first', model_path)
    exit()

saver.restore(sess, tf.train.latest_checkpoint(model_path))
graph = tf.get_default_graph()


def get_prediction(filename):
    image_path = image_dir_path + filename
    x_batch = get_images_formatted(image_path)

    y_prediction = graph.get_tensor_by_name("y_pred:0")

    x = graph.get_tensor_by_name("x:0")
    y_true = graph.get_tensor_by_name("y_true:0")
    y_test_images = np.zeros((1, 2))

    feed_dict_testing = {x: x_batch, y_true: y_test_images}
    result = sess.run(y_prediction, feed_dict=feed_dict_testing)

    is_dog_probability = result[0][0]
    is_cat_probability = result[0][1]
    if is_dog_probability >= is_cat_probability:
        print(filename + ' The captured image is a dog\n')
    else:
        print(filename + ' The captured image is a cat\n')


# Real time input

cap = cv2.VideoCapture(0)  # Get default video input
cap.set(3, 640)  # Set Width
cap.set(4, 480)  # Set Height

period = 5  # seconds
count = 0

# Find OpenCV version
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

# With webcam get(CV_CAP_PROP_FPS) does not work.
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
        image_name = ''.join(['image', str(count), '.jpg'])
        cv2.imwrite('real_time_images\\'+image_name, frame)
        last_capture = time.time()
        print(image_name + ' saved successfully')
        count += 1
        get_prediction(filename=image_name)

    # Stop Condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



