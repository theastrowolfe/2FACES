import os
import tensorflow as tf
from keras.models import Model
from keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
from keras.src.models.functional import Functional
from keras.metrics import Precision, Recall
from keras.saving import load_model, register_keras_serializable

"""
Model Engineering
"""

# Build siamese distance layer classes
@register_keras_serializable()
class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    # Similarity calculation (distance)
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding[0] - validation_embedding[0])

@register_keras_serializable()
class L2Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    # Similarity calculation
    def call(self, input_embedding, validation_embedding):
        sum_square = tf.math.reduce_sum(tf.math.square(input_embedding - validation_embedding), axis=1, keepdims=True)
        return tf.math.sqrt(sum_square, tf.keras.backend.epsilon())

@register_keras_serializable()
class L2Norm(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    # Similarity calculation
    def call(self, x):
        return x / tf.math.sqrt(tf.math.reduce_sum(tf.math.multiply(x, x), axis=1, keepdims=True))

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

# Produce the siamese neural network model
def make_siamese_model(model=None):
    embedding = make_embedding()

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

class SiameseModel:
    def __init__(self, path=None):
        self._model = self._load(path) if path else make_siamese_model()

    def save(self, path="SiameseModel.keras"):
        self._model.save(path)

    def _load(self, path):
        return load_model(
            path,
            custom_objects={
                "L1Dist": L1Dist,
                "BinaryCrossentropy": tf.losses.BinaryCrossentropy
            }
        )

    def predict(self, *args, **kwargs):
        return self._model.predict(*args, **kwargs)

    def train(self, data, EPOCHS=10, checkpoint_dir="models/checkpoints"):
        # Setup loss function
        binary_cross_loss = tf.losses.BinaryCrossentropy()

        # Setup optimizer (more at https://keras.io/api/optimizers/)
        optimizer = tf.keras.optimizers.Adam(1e-4)

        checkpoint_dir = "models/checkpoints"
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, siamese_model=self._model)

        # Build custom training step
        @tf.function
        def train_step(batch):
            with tf.GradientTape() as tape:
                # Anchor and positive/negative image
                x = batch[:2]
                # Label (1 or 0)
                y_true = batch[2]

                # Forward pass (prediction)
                y_pred = self._model(x, training=True)
                # Calculate loss
                loss = binary_cross_loss(y_true, y_pred)
            print(f"loss: {loss}")

            # Calculate gradient
            gradients = tape.gradient(loss, self._model.trainable_variables)

            # Calculate and propagate the new weights using adam's optimization
            # algorithm, a variant of gradient decent. This updates weights and applies
            # them to the siamese model
            grads_and_vars = list(zip(gradients, self._model.trainable_variables))
            optimizer.apply_gradients(grads_and_vars)

            return loss

        for epoch in range(0, EPOCHS+1):
            print(f"Epoch {epoch}/{EPOCHS}")

            # Create a metric object
            r = Recall()
            p = Precision()

            for _, batch in enumerate(data):
                loss = train_step(batch)
                y_true = self._model.predict(batch[:2])
                r.update_state(batch[2], y_true)
                p.update_state(batch[2], y_true)
            print(f"Loss: {loss.numpy()}, Recall: {r.result().numpy()}, Precision: {p.result().numpy()}")

            if epoch % 10 == 0:
                checkpoint.save(file_prefix=checkpoint_prefix)