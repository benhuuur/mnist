import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, accuracy_score
from joblib import load
import numpy as np

# df = pd.read_csv("mnist_train.csv")
df = pd.read_csv("mnist_test.csv")

df.dropna(inplace=True)

df = df[5000:10000]

Y = df["label"]
X = df.drop("label", axis=1)

loaded_pca = load("models/pca_model.pkl")
X = loaded_pca.transform(X)

loaded_model = load("models/svc_model.pkl")
print("Test Mean Absolute Error: ", mean_absolute_error(Y, loaded_model.predict(X)))
print("Test Accuracy: ", accuracy_score(Y, loaded_model.predict(X)))

predictions = loaded_model.predict(X)

plt.scatter(x=np.arange(Y.size), y=Y, s=20)
plt.scatter(x=np.arange(predictions.size), y=predictions, s=10)
plt.show()
