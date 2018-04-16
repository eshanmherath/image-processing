import cv2
import numpy as np

image_size = 128
num_channels = 3


def get_images_formatted(path):
    images = []
    image = cv2.imread(path)
    image = cv2.resize(image, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)
    images.append(image)
    images = np.array(images, dtype=np.uint8)
    images = images.astype('float32')
    images = np.multiply(images, 1.0 / 255.0)
    x_batch_temp = images.reshape(1, image_size, image_size, num_channels)
    return x_batch_temp
