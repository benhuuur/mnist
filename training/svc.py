import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, f1_score, mean_absolute_error

from joblib import dump

df = pd.read_csv("data/mnist_train.csv")

df.dropna(inplace=True)

df = df

Y = df["label"]
X = df.drop("label", axis=1)

# print("CV")
# scores = cross_val_score(SVC(), X, Y, cv=8)
# print("Scores by CV:",scores)

# print("PCA")
# pca = PCA(n_components=784)
# pca.fit(X)
# dump(pca, "models/pca/pca_model_all.pkl")

# X = pca.transform(X)

print("Split")
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.35, random_state=42
)

print("GridSearchCV")
model = GridSearchCV(
    SVC(),
    {
        "C": list(map(lambda x: x, range(1, 10))),
        "kernel": ["linear", "poly", "rbf"],
        # "degree": list(map(lambda x: x, range(1, 4))),
        "tol": list(map(lambda x: x / 1e5, range(1, 10))),
    },
    n_jobs=-1,
)

print("GridSearchCV Fit")
model.fit(X_train, Y_train)
print("SVC best params:", model.best_params_)

print("Evaluation")
Y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
precision = precision_score(Y_test, Y_pred, average='weighted')
f1 = f1_score(Y_test, Y_pred, average='weighted')
print("Accuracy:", accuracy)
print("Precision:", precision)
print("F1 Score:", f1)
print("Mean Absolute Error:", mean_absolute_error(Y_test, Y_pred))

print("Save best model")
model = model.best_estimator_
dump(model, "models/algorithm/svc_model_all.pkl")