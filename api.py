from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
import cv2
from io import BytesIO
from PIL import Image

app = FastAPI(title="Freshness Classifier API")

# Load the model once when the server starts
model = tf.keras.models.load_model("model/freshness_model.h5")

IMG_SIZE = (224, 224)

def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
    return img_array

@app.get("/")
def home():
    return {"message": "Freshness Classifier API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents))
    
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)[0][0]
    
    label = "rotten" if prediction > 0.5 else "fresh"
    confidence = float(prediction) if label == "rotten" else float(1 - prediction)
    
    return JSONResponse({
        "prediction": label,
        "confidence": round(confidence, 4)
    })