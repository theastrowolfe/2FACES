from keras.metrics import Precision, Recall
from matplotlib import pyplot as plt

from data import test_data, train_data
from model import SiameseModel

import uuid

# Create new model
siamese_model = SiameseModel()
siamese_model.train(train_data, 50)
siamese_model.save(f"models/{uuid.uuid1()}.keras")

# Load existing model
#siamese_model = SiameseModel("models/d361ff92-e67f-11ee-8323-e335da2c34e8.keras")

r = Recall()
p = Precision()
for test_input, test_val, y_true in test_data.as_numpy_iterator():
    # Test making predictions
    y_pred = siamese_model.predict([test_input, test_val])

    # Post process results
    print([1 if prediction > 0.5 else 0 for prediction in y_pred])

    # Create metric object for recall and calculate recall value
    r.update_state(y_true, y_pred)

    # Create metric object for precision and calculate precision value
    p.update_state(y_true, y_pred)
print(f"Recall: {r.result().numpy()}")
print(f"Precision: {p.result().numpy()}")

"""
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.imshow(test_input[3])
plt.subplot(1, 2, 2)
plt.imshow(test_val[3])
plt.show()
"""
