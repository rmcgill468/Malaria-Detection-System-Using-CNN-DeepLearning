import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("malaria_model.h5")

IMG_SIZE = 64

def predict_image(image_path):

    img = cv2.imread(image_path)
    img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
    img = img/255.0
    img = np.reshape(img,[1,IMG_SIZE,IMG_SIZE,3])

    prediction = model.predict(img)

    if prediction > 0.5:
        print("Uninfected Cell")
    else:
        print("Parasitized Cell")

predict_image("test.jpg")
