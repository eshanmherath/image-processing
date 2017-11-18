import cv2
from matplotlib import pyplot as plt

test_image = cv2.imread('superman_test.jpg', 0)
print('\n Matching test image "superman_test.jpg"')

train_image_names = ['superman.jpg', 'batman.jpg', 'hulk.jpg']
train_images = [None, None, None]
for i in range(len(train_image_names)):
    train_images[i] = cv2.imread(train_image_names[i], 0)

for i in range(len(train_images)):
    # Initiate SIFT detector
    orb = cv2.ORB_create()

    # find the key points and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(test_image, None)
    kp2, des2 = orb.detectAndCompute(train_images[i],None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1,des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    # visualize the matches. Draw first 100 matches.
    img3 = cv2.drawMatches(test_image, kp1, train_images[i], kp2, matches[:100], outImg=None, flags=2)
    plt.imshow(img3), plt.show()

    dist = [m.distance for m in matches]

    # threshold: half the mean
    threshold_distance = (sum(dist) / len(dist)) * 0.5
    # keep only the reasonable matches
    selected_matches = [m for m in matches if m.distance < threshold_distance]
    print('Train image "' + str(train_image_names[i]) + '" Similarity Score :' + str(len(selected_matches)))

