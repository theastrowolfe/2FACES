#!/usr/bin/env python3

"""
Author: Joshua Wolfe

The model being built is a siamese neural network to do one-shot
classification. Two imputs to the input layer and output a singular value, one
1 for verified and 0 for unverified.

NOTE: Proof of concept!

source: https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf
"""

"""
SETUP
"""

# Standard imports
import functools
import os
import uuid

# 3rd party imports
import cv2
import tensorflow as tf
from matplotlib import pyplot as plt

# Functional Tensorflow imports
from keras.models import Model
from keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
from keras.metrics import Precision, Recall

"""
Model - Allows us to compile the Layer and Input together
Layer - Allows us to define a custom neural network layer and effectively create a custom class to generate layers
Input - Allows us to define what is passed through to models
Flatten - Effectively flattens all the information form previous layers and flatten it down and pass convolutional neural network data to a dense layer
Conv2D - Allows us to perform convolutions
Dense - Allows us to create a fully connected layer, typical of most neural networks
MaxPooling2D - Max values of a particular region that gets passed from one layer to the next
"""

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

# Move LFW images to negative images
# Note: images are 250x250 pixels
for directory in os.listdir(RAW_PATH):
    for file in os.listdir(os.path.join(RAW_PATH, directory)):
        old_path = os.path.join(RAW_PATH, directory, file)
        new_path = os.path.join(NEG_PATH, file)
        os.replace(old_path, new_path)

# Connect to webcam (1 webcam so 0 based counting?)
cap = cv2.VideoCapture(0)

anc_complete = len(os.listdir(ANC_PATH)) == 400
pos_complete = len(os.listdir(POS_PATH)) == 400
# Capture images
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
    elif cv2.waitKey(1) & 0XFF == ord('q'): # Gracefully break
        break
# Clean up
cap.release()
cv2.destroyAllWindows()

"""
Load and Preprocess Images
"""

# Get data
anchor = tf.data.Dataset.list_files(ANC_PATH+'/*.jpg').take(300)
positive = tf.data.Dataset.list_files(POS_PATH+'/*.jpg').take(300)
negative = tf.data.Dataset.list_files(NEG_PATH+'/*.jpg').take(300)

# Resize and scale
def preprocess(path):
    byte_img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(byte_img)
    img = tf.image.resize(img, (100, 100))
    img = img / 255.0
    return img

# Create Labeled Dataset of tupples (anchor_image, pos/neg_image, 1/0)
positives =  tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negatives =  tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
data = positives.concatenate(negatives)

# Build train and test partition
def preprocess_twin(input_img, validation_img, label):
    return (preprocess(input_img), preprocess(validation_img), label)

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

"""
Model Engineering
"""

# Build an embedding layer
def make_embedding():
    input_layer = Input(shape=(100, 100, 3), name="input_image")

    # First block
    c1 = Conv2D(64, (10, 10), activation="relu", name="convo1")(input_layer)
    m1 = MaxPooling2D(64, (2, 2), padding="same", name="max_pooling1")(c1)

    # Second block
    c2 = Conv2D(128, (7, 7), activation="relu", name="convo2")(m1)
    m2 = MaxPooling2D(64, (2, 2), padding="same", name="max_pooling2")(c2)

    # Third block
    c3 = Conv2D(128, (4, 4), activation="relu", name="convo3")(m2)
    m3 = MaxPooling2D(64, (2, 2), padding="same", name="max_pooling3")(c3)

    # Final embedding block
    c4 = Conv2D(256, (4, 4), activation="relu", name="convo4")(m3)
    f1 = Flatten(name="feature_maps")(c4)
    d1 = Dense(4096, activation="sigmoid", name="feature_vector")(f1)

    return Model(inputs=[input_layer], outputs=[d1], name="embedding")

embedding = make_embedding()
embedding.summary()

# Build siamese distance layer classes
class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    # Similarity calculation (distance)
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding[0] - validation_embedding[0])

class L2Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    # Similarity calculation
    def call(self, input_embedding, validation_embedding):
        sum_square = tf.math.reduce_sum(tf.math.square(input_embedding - validation_embedding), axis=1, keepdims=True)
        return tf.math.sqrt(sum_square, tf.keras.backend.epsilon())

class L2Norm(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    # Similarity calculation
    def call(self, x):
        return x / tf.math.sqrt(tf.math.reduce_sum(tf.math.multiply(x, x), axis=1, keepdims=True))

# Produce the siamese neural network model
def make_siamese_model(model=None):
    # Anchor image input in the network
    input_image = Input(name="input_image", shape=(100, 100, 3))

    # Validation image input in the network
    valid_image = Input(name="valid_image", shape=(100, 100, 3))

    # Combine siamese distance components
    siamese_layer = L1Dist()
    siamese_layer._name = "distance"
    if model:
        norm1 = L2Norm()(model(input_image))
        norm2 = L2Norm()(model(valid_image))
        distances = siamese_layer(norm1, norm2)
    else:
        input_embedding = embedding(input_image)
        valid_embedding = embedding(valid_image)
        distances = siamese_layer(input_embedding, valid_embedding)

    # Classifier layer
    classifier = Dense(1, activation="sigmoid", name="classifier")(distances)

    return Model(inputs=[input_image, valid_image], outputs=classifier, name="SiameseNetwork")

siamese_model = make_siamese_model()
siamese_model.summary()

"""
Training
"""

# Setup loss function
binary_cross_loss = tf.losses.BinaryCrossentropy()

# Setup optimizer (more at https://keras.io/api/optimizers/)
optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = "models/checkpoints"
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, siamese_model=siamese_model)

# Build custom training step
@tf.function
def train_step(batch):
    with tf.gradienttape() as tape:
        # anchor and positive/negative image
        x = batch[:2]
        # label (1 or 0)
        y_true = batch[2]

        # forward pass (prediction)
        y_pred = siamese_model(x, training=True)
        # calculate loss
        loss = binary_cross_loss(y_true, y_pred)
    print(f"loss: {loss}")

    # calculate gradient
    gradients = tape.gradient(loss, siamese_model.trainable_variables)

    # calculate and propagate the new weights using adam's optimization
    # algorithm, a variant of gradient decent. this updates weights and applies
    # them to the siamese model
    grads_and_vars = list(zip(gradients, siamese_model.trainable_variables))
    optimizer.apply_gradients(grads_and_vars)

    return loss

# Create a training loop
def train(data, EPOCHS=0):
    for epoch in range(0, EPOCHS+1):
        print(f"Epoch {epoch}/{EPOCHS}")

        # Create a metric object
        r = Recall()
        p = Precision()

        for idx, batch in enumerate(data):
            loss = train_step(batch)
            y_true = siamese_model.predict(batch[:2])
            r.update_state(batch[2], y_true)
            p.update_state(batch[2], y_true)
        print(f"Loss: {loss.numpy()}, Recall: {r.result().numpy()}, Precision: {p.result().numpy()}")

        if epoch % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

# Train the model
EPOCHS = 50
train(train_data, EPOCHS)

#siamese_model.save("siamesemodel_50.keras")
siamese_model.summary()


Evaluate Model

siamese_model.save(f"siamesemodel_{EPOCHS}.keras")
siamese_model.summary()

test_input, test_value, y_true = test_data.as_numpy_iterator().next()
y_pred = siamese_model.predict([test_input, test_value])
print([1 if prediction > 0.5 else 0 for prediction in y_pred])
print(y_true)

x = Recall()
x.update_state(y_true, y_pred)
print(f"Recall result: {x.result().numpy()}")

x = Precision()
x.update_state(y_true, y_pred)
print(f"Precision result: {x.result().numpy()}")

r = Recall()
p = Precision()
for test_input, test_val, y_true in test_data.as_numpy_iterator():
    y_pred = siamese_model.predict([test_input, test_value])
    r.update_state(y_true, y_pred)
    p.update_state(y_true, y_pred)
print(f"Recall: {r.result()}, Precision: {p.result().numpy()}")

siamese_model.save("siamesemodel.h5")
siamese_model.summary()
