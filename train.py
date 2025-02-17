import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
import skops.io as sio


data = pd.read_csv("./Data/drug.csv")
data = data.sample(frac=1, random_state=42)

from sklearn.model_selection import train_test_split

X = data.drop("Drug", axis=1).values
y = data.Drug.values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

transformer = ColumnTransformer(
    transformers=[
        ("encoder", OrdinalEncoder(), [1, 2, 3]),
        ("imputer", SimpleImputer(strategy="median"), [0, 4]),
        ("scaler", StandardScaler(), [0, 4]),
    ]
)

pipe = Pipeline(
    steps=[
        ("preprocessor", transformer),
        ("model", RandomForestClassifier(n_estimators=10, random_state=42)),
    ]
)

pipe.fit(X_train, y_train)

predictions = pipe.predict(X_test)

accu = accuracy_score(y_test, predictions)
print(accu)
f1 = f1_score(y_test, predictions, average="macro")
print(f"accuracy: {round(accu, 2)} f1: {round(f1, 2)}")

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

preds = pipe.predict(X_test)
cm = confusion_matrix(y_test, preds, labels=pipe.classes_)
disp = ConfusionMatrixDisplay(cm, display_labels=pipe.classes_)
disp.plot()
plt.savefig("./Results/model_results.png", dpi=120)

with open("./Results/metrics.txt", "w") as f:
    f.write(f"accuracy: {round(accu, 2)} f1: {round(f1, 2)}")

sio.dump(pipe, "./Model/drug_pipeline.skops")

import joblib

# Save your trained model
joblib.dump(pipe, "model.pkl")
