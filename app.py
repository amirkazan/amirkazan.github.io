# -*- coding: utf-8 -*-
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import json
from flask import jsonify
import sys
import codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())



def preprocess_image(file):
    image = Image.open(file)
    # Преобразование изображения в формат RGB
    image = image.convert('RGB')
    
    # Загрузка изображения и изменение размера
    img = image.resize((180, 180))
    
    # Преобразование изображения в тензор
    img_array = tf.keras.utils.img_to_array(img)
    
    # Добавление измерения батча
    img_tensor = tf.expand_dims(img_array, 0)

    return img_tensor



app = Flask(__name__)
model = load_model('test_model.keras')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    # Обработка изображения перед передачей его в модель
    # Пример: предобработка изображения для модели
    image = preprocess_image(file)
    # Получение предсказания от модели
    prediction = model.predict(image)
    # Отправка ответа клиенту
    return jsonify(prediction.tolist())


if __name__ == '__main__':
    app.run(debug=True)

