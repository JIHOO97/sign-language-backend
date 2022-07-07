from fastapi import FastAPI, File
from prediction import processImage, predict
app = FastAPI()

@app.get('/index')
def hello():
    return 'hello world'

@app.post('/api/predict')
def predict_image(file: bytes = File(...)):
    image = processImage(file)
    result = predict(image)
    return result