import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, f1_score, mean_absolute_error
from joblib import dump

df = pd.read_csv("data/mnist_train.csv")

df.dropna(inplace=True)

Y = df["label"]
X = df.drop("label", axis=1)

print("CV")
scores = cross_val_score(SGDClassifier(), X, Y, cv=3)
print("Scores by CV:", scores)

print("Split")
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.35, random_state=42
)

print("GridSearchCV")
model = GridSearchCV(
    SGDClassifier(),
    {
        "random_state": [42],
        "alpha": [0.0001, 0.001, 0.01],
        "max_iter": [1000, 2000, 3000],
        "loss": ['hinge', 'log', 'modified_huber']
    },
    n_jobs=-1,
)

print("GridSearchCV Fit")
model.fit(X_train, Y_train)
print("SGDClassifier best params:", model.best_params_)

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
dump(model, "models/algorithm/sgd_model.pkl")
