import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error
from joblib import dump

df = pd.read_csv("data/mnist_train.csv")

df.dropna(inplace=True)

# df = df[5000:10000]

Y = df["label"]
X = df.drop("label", axis=1)

print("CV")
scores = cross_val_score(RandomForestClassifier(), X, Y, cv=8)
print("Scores by CV:", scores)

print("PCA")
pca = PCA(n_components=784)
pca.fit(X)
dump(pca, "models/pca/pca_model_RFC.pkl")

X = pca.transform(X)

print("Split")
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.35, random_state=42
)

print("GridSearchCV")
model = GridSearchCV(
    RandomForestClassifier(),
    {
        "random_state": [42],
        "n_estimators": [50, 100, 200],
        "criterion": ["gini", "entropy"]
    },
    n_jobs=-1,
)

print("GridSearchCV Fit")
model.fit(X_train, Y_train)
print("SVC best params:", model.best_params_)

print("Mean Absolute Error: ", mean_absolute_error(Y, model.predict(X)))

print("Save best model")
model = model.best_estimator_
dump(model, "models/algorithm/rfc_model.pkl")
