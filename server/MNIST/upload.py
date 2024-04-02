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
    # fazer 
    loaded_pca = load("../models/PCA/pca_model_all.pkl")
    rfc_model = load("../models/algorithm/rfc_model_allPCA_flask.pkl")
    svc_model = load("../models/algorithm/svc_model_allPCA_flask.pkl")
    knn_model = load("../models/algorithm/KNN_model_flask.pkl")
    linear_svc_model = load("../models/algorithm/linear_svc_model_flask.pkl")
    if(len(X) > loaded_pca.n_components_):
        X = X[1:] 
    X = pd.DataFrame([X])
    predictions =[]
    for i in range(10):
        predictions.append(0);
    
    print(X)

    predict = knn_model.predict(X)[0]
    print("knn:", predict)
    predictions[predict] += 1.2

    predict = linear_svc_model.predict(X)[0]
    print("linear_svc_model:", predict)
    predictions[predict] += 1
    
    X = loaded_pca.transform(X)

    predict = rfc_model.predict(X)[0]
    print("rfc_model:", predict)
    predictions[predict] += 1
    
    predict = svc_model.predict(X)[0]
    print("svc_model:", predict)
    predictions[predict] += 1.1
    
    for i in range(len(predictions)):
        if predictions[i] == max(predictions):
            return [i]
    return [None]
