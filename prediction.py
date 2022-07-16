from PIL import Image, ImageOps
from io import BytesIO
import tensorflow as tf
import numpy as np

# path config
MODEL_PATH = 'model/savedmodel'

LABEL_PATH = 'model/labels.txt'

# load models
model = tf.keras.models.load_model(MODEL_PATH)

# load labels
def read_label(LABEL_PATH):
    labels = {}
    with open(LABEL_PATH) as f:
        lines = f.readlines()
        for line in lines:
            idx, label = line.split()
            labels[int(idx)] = label
    return labels

label = read_label(LABEL_PATH)

# image processing
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
size = (224, 224)

# image processing
def processImage(encoded_img):
    image = Image.open(BytesIO(encoded_img))
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    return data

# prediction
def predict(data):
    predictions = []
    prediction = model.predict(data)
    label_map = {}
    for idx, score in enumerate(prediction[0]):
        label_map[label[idx]] = np.around(score * 100, decimals=2)
    predictions.append(label_map)
    return predictions