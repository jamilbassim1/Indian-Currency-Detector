from django.shortcuts import render
import tensorflow as tf
import numpy as np
import cv2, os

# load model once
model = tf.keras.models.load_model("currency_model.h5")
labels = ['10','20','50','100','200','500','2000']

def predict_currency(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224,224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    index = np.argmax(pred)

    return labels[index]

def upload(request):
    if request.method == 'POST':
        image = request.FILES['image']

        upload_path = os.path.join("static/uploads", image.name)

        with open(upload_path, 'wb+') as dest:
            for chunk in image.chunks():
                dest.write(chunk)

        result = predict_currency(upload_path)

        return render(request, 'result.html', {
            "result": result,
            "image_url": "/media/" + image.name
        })

    return render(request, 'upload.html')
