import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV
from joblib import dump
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, f1_score, mean_absolute_error

df = pd.read_csv("data/mnist_train.csv")

df.dropna(inplace=True)

Y = df["label"]
X = df.drop("label", axis=1)

print("Split")
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.35, random_state=42
)

print("GridSearchCV")
model = GridSearchCV(
    LinearSVC(),
    {
        "random_state": [42],
        "C": [0.1, 1, 10],
        "dual": [False], 
    },
    n_jobs=-1,
)

print("GridSearchCV Fit")
model.fit(X_train, Y_train)
print("LinearSVC best params:", model.best_params_)

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
best_model = model.best_estimator_
dump(best_model, "models/algorithm/linear_svc_model.pkl")
