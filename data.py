import cv2
import os
import uuid
import tensorflow as tf
from matplotlib import pyplot as plt

# Constants
VOFFSET = 20
HOFFSET = 220
HEIGHT = 250+VOFFSET
WIDTH = 250+HOFFSET

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

"""
POSITIVE AND NEGATIVE DATA COLLECTION
"""

# Data folder structure
RAW_PATH = os.path.join("data", "raw", "lfw")
ANC_PATH = os.path.join("data", "anchor")
POS_PATH = os.path.join("data", "positive")
NEG_PATH = os.path.join("data", "negative")
LIMIT = min(
    len(os.listdir(ANC_PATH)),
    len(os.listdir(POS_PATH)),
    len(os.listdir(NEG_PATH))
)

# Move LFW images to negative images
# Note: images are 250x250 pixels
for directory in os.listdir(RAW_PATH):
    for file in os.listdir(os.path.join(RAW_PATH, directory)):
        old_path = os.path.join(RAW_PATH, directory, file)
        new_path = os.path.join(NEG_PATH, file)
        os.replace(old_path, new_path)

# Connect to webcam (1 webcam so 0 based counting?)
cap = cv2.VideoCapture(0)

# Capture images
anc_complete = len(os.listdir(ANC_PATH)) == LIMIT
pos_complete = len(os.listdir(POS_PATH)) == LIMIT
while not anc_complete and not pos_complete and cap.isOpened():
    ret, frame = cap.read()

    # Trim frame to 250x250px with offset to center in webcam
    frame = frame[VOFFSET:HEIGHT, HOFFSET:WIDTH, :]
    cv2.imshow("Image Collection", frame)

    if cv2.waitKey(1) & 0XFF == ord('a'): # Collect anchors
        if anc_complete:
            print("ANCHOR DONE")
            continue
        imgname = os.path.join(ANC_PATH, f"{uuid.uuid1()}.jpg")
        cv2.imwrite(imgname, frame)
        anc_complete = len(os.listdir(ANC_PATH)) == 400
    elif cv2.waitKey(1) & 0XFF == ord('p'): # Collect positives
        if pos_complete:
            print("POSITIVE DONE")
            continue
        imgname = os.path.join(POS_PATH, f"{uuid.uuid1()}.jpg")
        cv2.imwrite(imgname, frame)
        pos_complete = len(os.listdir(POS_PATH)) == 400
    if cv2.waitKey(1) & 0XFF == ord('q'): # Gracefully break
        break
# Clean up
cap.release()
cv2.destroyAllWindows()

"""
Load and Preprocess Images
"""

# Get data
anchor   = tf.data.Dataset.list_files(ANC_PATH+'/*.jpg').take(LIMIT)
positive = tf.data.Dataset.list_files(POS_PATH+'/*.jpg').take(LIMIT)
negative = tf.data.Dataset.list_files(NEG_PATH+'/*.jpg').take(LIMIT)

# Create Labelled Dataset of tupples (anchor_image, pos/neg_image, 1/0)
positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
data = positives.concatenate(negatives)

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

# Resize and scale
def preprocess_file(path):
    return preprocess_image(get_image_data(path))

# Build train and test partition
def preprocess_twin(input_img, validation_img, label):
    return (preprocess_file(input_img), preprocess_file(validation_img), label)

# Data pipeline
data = data.map(preprocess_twin)
data = data.cache()
data = data.shuffle(buffer_size=1024)

# Training partition
train_data = data.take(round(len(data) * 0.7)) # Effectively take 70% of the data to train on
train_data = train_data.batch(16) # Pass through data as batches of 16
train_data = train_data.prefetch(8) # Effectively starts preprocessing to improve training and prevent a bottleneck

# Validation partition
test_data = data.skip(round(len(data) * 0.7))
test_data = test_data.take(round(len(data) * 0.3))
test_data = test_data.batch(16)
test_data = test_data.prefetch(8)