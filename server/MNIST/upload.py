from flask import Blueprint, render_template, request
from PIL import Image
import pandas as pd
from joblib import load

bp = Blueprint('auth', __name__, url_prefix='/')

@bp.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        imagem = request.files['imagem']
        imagem.save('../data/images/test.png')
        path_to_image = '../data/images/test.png'
        gray_vector = get_vector_from_image(path_to_image)
        predict =  get_prediction(gray_vector)
        predict_str = ', '.join(str(prediction) for prediction in predict)
        return predict_str
    return render_template('index.html')


def get_vector_from_image(path):
    image = Image.open(path)
    width, height = image.size

    gray_vector = []

    for y in range(height):
        for x in range(width):
            pixel = image.getpixel((x, y))
            intensity = 255 - int(0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2])
            gray_vector.append(intensity)

    return gray_vector

def get_prediction(X):
    loaded_pca = load("../models/pca_model_allPCA.pkl")
    loaded_model = load("../models/svc_model_allPCA.pkl")
    if(len(X) > loaded_pca.n_components_):
        X = X[1:] 
    X = pd.DataFrame([X])
    X = loaded_pca.transform(X)
    predictions = loaded_model.predict(X)
    print(predictions)
    return predictions