from fastapi import FastAPI
from fastapi import File, UploadFile
import os
import pickle
from tensorflow import keras, expand_dims
import numpy as np
from PIL import Image
import io

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))

docs_url = "/docs"

app = FastAPI(docs_url=docs_url, debug=True)

@app.get("/")
def read_root():
    return {"Alzheimer_prediction": "v1.0.0"}

@app.post("/detect_alzheimer")
def upload(file: UploadFile):

    image = file.file.read()
    a = read_imagefile(image)
    

    img_width, img_height = 176, 176
    img = keras.utils.load_img(
        a, target_size=(img_height, img_width)
    )
    img_array = keras.utils.img_to_array(img)
    img_array = expand_dims(img_array, 0)

    # load model

    MODEL_PATH = os.path.join(ROOT_DIR, "app", "my_model_v2.h5")
    model = keras.models.load_model(MODEL_PATH)
   
    #predict
    CLASSES = [ 'MildDemented',
        'ModerateDemented',
        'NonDemented',
        'VeryMildDemented']

    predictions = model.predict(img_array)

    ll = (predictions[0].tolist())

    max_index = ll.index(max(ll))
    return (CLASSES[max_index])

def read_imagefile(file) -> Image.Image:
    image = (io.BytesIO(file))
    print(image)
    return image
