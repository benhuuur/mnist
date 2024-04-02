import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, f1_score, mean_absolute_error
from joblib import dump

df = pd.read_csv("data/mnist_train.csv")

df.dropna(inplace=True)

Y = df["label"]
X = df.drop("label", axis=1)

print("Cross-validation")
scores = cross_val_score(KNeighborsClassifier(), X, Y, cv=8)
print("Scores by CV:", scores)
print("Mean Accuracy:", scores.mean())

# PCA
# print("PCA")
# pca = PCA(n_components=784)
# pca.fit(X)
# dump(pca, "models/pca/pca_model_KNG.pkl")
# X_pca = pca.transform(X)

print("Split")
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.35, random_state=42
)

# GridSearchCV
print("GridSearchCV")
param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'p' : [1, 2],
    'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute']
    } 

model = GridSearchCV(
    KNeighborsClassifier(),
    param_grid,
    n_jobs=-1,
)

print("GridSearchCV Fit")
model.fit(X_train, Y_train)
print("Best params:", model.best_params_)

print("Evaluation")
Y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
precision = precision_score(Y_test, Y_pred, average='weighted')
f1 = f1_score(Y_test, Y_pred, average='weighted')
print("Accuracy:", accuracy)
print("Precision:", precision)
print("F1 Score:", f1)
print("Mean Absolute Error:", mean_absolute_error(Y_test, Y_pred))

print("Saving the best model")
best_model = model.best_estimator_
dump(best_model, "models/algorithm/KNG_model.pkl")
