import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score
from joblib import load
import numpy as np
import json

# loaded_pca = load("models/pca/pca_model_rfc.pkl")
loaded_model = load("models/algorithm/KNG_model.pkl")

# from csv
df = pd.read_csv("data/mnist_train.csv")
df = pd.read_csv("data/mnist_test.csv")

df.dropna(inplace=True)

Y = df["label"]
X = df.drop("label", axis=1)

# X = loaded_pca.transform(X)

predictions = loaded_model.predict(X)
accuracy = accuracy_score(Y, predictions)
print("Acurácia do modelo nos dados de teste:", accuracy)
precision = precision_score(Y, predictions, average='weighted')
print("Precisão do modelo nos dados de teste:", precision)

plt.scatter(x=np.arange(Y.size), y=Y, s=20)
plt.scatter(x=np.arange(predictions.size), y=predictions, s=10)
plt.show()


# from image
with open('data\\Row.json', 'r') as arquivo:
    X = json.load(arquivo)
    
# if(len(X) > loaded_pca.n_components_):
#     X = X[1:] 
X = pd.DataFrame([X])
# X = loaded_pca.transform(X)
predictions = loaded_model.predict(X)
print(predictions)