from PIL import Image, ImageOps
from io import BytesIO
import numpy as np
import tensorflow as tf

# path config
MODEL1_PATH = 'models/ILY_HOME_IRR_CARE_H_RIGHT/model1'
MODEL2_PATH = 'models/FINE_WHY_WAIT/model2'

LABEL1_PATH = 'models/ILY_HOME_IRR_CARE_H_RIGHT/labels.txt'
LABEL2_PATH = 'models/FINE_WHY_WAIT/labels.txt'

# load models
model1 = tf.keras.models.load_model(MODEL1_PATH)
model2 = tf.keras.models.load_model(MODEL2_PATH)

# load labels
def read_label(LABEL_PATH):
    labels = {}
    with open(LABEL_PATH) as f:
        lines = f.readlines()
        for line in lines:
            idx, label = line.split()
            labels[int(idx)] = label
    return labels

label1 = read_label(LABEL1_PATH)
label2 = read_label(LABEL2_PATH)

models = [model1, model2]
labels = [label1, label2]

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
    for i in range(len(models)):
        prediction = models[i].predict(data)
        label_map = {}
        for idx, score in enumerate(prediction[0]):
            label_map[labels[i][idx]] = np.around(score * 100, decimals=2)
        predictions.append(label_map)
    return predictions