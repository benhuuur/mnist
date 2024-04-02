import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score, mean_absolute_error
from joblib import dump, load

df = pd.read_csv("data/mnist_train.csv")

df.dropna(inplace=True)

Y = df["label"]
X = df.drop("label", axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.35, random_state=42
)

rf_model = load("models/algorithm/rfc_model_allPCA.pkl")
svc_model = load("models/algorithm/svc_model_allPCA.pkl")
knn_model = load("models/algorithm/KNG_model.pkl")
linear_svc_model = load("models/algorithm/linear_svc_model.pkl")
sgd_model = load("models/algorithm/sgd_model.pkl")

voting_model = VotingClassifier(estimators=[
    ('rf', rf_model),
    ('svc', svc_model),
    ('knn', knn_model),
    ('linear_svc', linear_svc_model),
    ('sgd', sgd_model)
], voting='hard')

print("Evaluation")
voting_model.fit(X_train, Y_train)
Y_pred = voting_model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
precision = precision_score(Y_test, Y_pred, average='weighted')
f1 = f1_score(Y_test, Y_pred, average='weighted')
print("Accuracy:", accuracy)
print("Precision:", precision)
print("F1 Score:", f1)
print("Mean Absolute Error:", mean_absolute_error(Y_test, Y_pred))

print("Saving the voting model")
dump(voting_model, "models/algorithm/VotingClassifier_model.pkl")
