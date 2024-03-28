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
        return render_template('result.html', predict_str=predict_str)
    return render_template('index.html', predict_str=None)


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
    loaded_pca = load("../models/PCA/pca_model_all.pkl")
    rfc_model = load("../models/algorithm/rfc_model_allPCA.pkl")
    svc_model = load("../models/algorithm/svc_model_allPCA.pkl")
    kng_model = load("../models/algorithm/KNG_model.pkl")
    if(len(X) > loaded_pca.n_components_):
        X = X[1:] 
    X = pd.DataFrame([X])
    predictions =[]
    for i in range(10):
        predictions.append(0);
    
    print(X)

    predictions[kng_model.predict(X)[0]] += 1.2
    X = loaded_pca.transform(X)
    predictions[rfc_model.predict(X)[0]] += 1
    predictions[svc_model.predict(X)[0]] += 1.1
    
    print(predictions)
    for i in range(len(predictions)):
        if predictions[i] == max(predictions):
            return [i]
    return [10]
