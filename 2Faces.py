import cv2
import os
import numpy as np
import tensorflow as tf

from model import SiameseModel
from matplotlib import pyplot as plt
from tensorflow.python.ops.image_ops_impl import ResizeMethod

model = SiameseModel("models/d361ff92-e67f-11ee-8323-e335da2c34e8.keras")

def get_image_data(path):
    byte_img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(byte_img)
    return img

def display(img):
    plt.figure(figsize=(5,5))
    plt.subplot(1,1,1)
    plt.imshow(img)
    plt.show()

def crop_to_face(img):
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Load the cascade
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

    # Detect faces
    #faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    faces = face_cascade.detectMultiScale(gray, 1.05, 6, minSize=[30,30])

    # Draw rectangle around the faces and crop the faces
    for (x, y, w, h) in faces:
        faces = img[y:y + h, x:x + w]
        return faces
    return []

def resize_image(img):
    return tf.image.resize(img, (100, 100))

def scale_image(img):
    return img / 255.0

def preprocess_image(img):
    return scale_image(resize_image(img))

def preprocess_file(path):
    return preprocess_image(get_image_data(path))

def verify(model, input_image, detect_threshold, verify_threshold):
    """
    Detection threshold: Metric above which a prediction is considered positive
    Verification threshold: Proportion of positive predictions / total positive samples
    """

    input_image_processed = preprocess_image(input_image)
    verification_path = os.path.join("data", "application_data", "verification_images")
    verification_images = os.listdir(verification_path)

    results = []
    for verification_image_path in verification_images:
        valid_image_processed = preprocess_file(os.path.join(verification_path, verification_image_path))

        results.append(
            model.predict(
                list(np.expand_dims([input_image_processed, valid_image_processed], axis=1))
            )
        )

    detection = np.sum(np.array(results) > detect_threshold)
    verification = detection / len(os.listdir(verification_path))
    verified = verification > verify_threshold

    return verified, verification, results

def capture():
    # Constants
    VOFFSET = 20
    HOFFSET = 110
    HEIGHT = 450+VOFFSET
    WIDTH = 450+HOFFSET

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()

        # Trim frame to 250x250px with offset to center in webcam
        frame = frame[VOFFSET:HEIGHT, HOFFSET:WIDTH, :]
        cv2.imshow("Image Collection", frame)

        if cv2.waitKey(1) & 0XFF == ord('q'): # Gracefully break
            print("QUITTING")
            break
        elif cv2.waitKey(1) & 0xFF == ord('c'): # Capture image
            print("CAPTURING")
            result = verify(model, frame, 0.01, 0.5)
            print(result)

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

capture()

"""
input_image = get_image_data("data/application_data/input_images/new0.jpg")
print(verify(model, input_image, 0.5, 0.5))
input_image = get_image_data("data/application_data/input_images/new1.jpg")
print(verify(model, input_image, 0.5, 0.5))
input_image = get_image_data("data/application_data/input_images/new2.jpg")
print(verify(model, input_image, 0.5, 0.5))
input_image = get_image_data("data/application_data/input_images/new3.jpg")
print(verify(model, input_image, 0.5, 0.5))
input_image = get_image_data("data/application_data/input_images/new4.jpg")
print(verify(model, input_image, 0.5, 0.5))
"""

#results, verified = verify(SiameseModel("models/97c7f612-e61b-11ee-8323-e335da2c34e8.keras"), 0.5, 0.5)
#print(results, verified)
#results, verified = verify(SiameseModel("models/f5ce3782-e632-11ee-8323-e335da2c34e8.keras"), 0.5, 0.5)
#print(results, verified)

#application_path = os.path.join("data", "application_data")
#input_image_path = os.path.join(application_path, "input_images", "input_image2.jpg")
#cv2.imwrite(input_image_path, frame)
"""
plt.figure(figsize=(10,8))
plt.subplot(3,1,1)
plt.imshow(frame)
plt.subplot(3,1,2)
plt.imshow(_crop_to_face(frame))
plt.subplot(3,1,3)
plt.imshow(_preprocess(_crop_to_face(frame)))
plt.show()
"""

"""
def verify(model, detection_threshold, verification_threshold):
    # Detection threshold: Metric above which a prediction is considered positive
    # Verification threshold: Proportion of positive predictions / total positive samples

    results = []
    application_path = os.path.join("data", "application_data")
    verification_path = os.path.join(application_path, "verification_images")
    for image in os.listdir(verification_path):
        valid_image = _preprocess(os.path.join(verification_path, image))
        input_image = _preprocess(os.path.join(application_path, "input_images", "input_image.jpg"))

        # Make prediction and store the result
        results.append(
            model.predict(
                list(np.expand_dims([input_image, valid_image], axis=1))
            )
        )

    detection = np.sum(np.array(results) > detection_threshold)
    verification = detection / len(os.listdir(verification_path))
    verified = verification > verification_threshold

    return results, verified
"""
