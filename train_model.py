import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import shap
import joblib
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("survey_lung_cancer.csv")

# Convert YES/NO values to 1/0
data.replace({"YES": 1, "NO": 0}, inplace=True)

# Convert GENDER column to numbers
data["GENDER"] = data["GENDER"].replace({"M": 1, "F": 0})

# Features and target
X = data.drop("LUNG_CANCER", axis=1)
y = data["LUNG_CANCER"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model using :contentReference[oaicite:0]{index=0}
model = XGBClassifier(
    n_estimators=250,
    max_depth=6,
    learning_rate=0.1
)

model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))

# Save model
joblib.dump(model, "model.pkl")

# Explainable AI using :contentReference[oaicite:1]{index=1}
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, show=False)
plt.savefig("shap_feature_importance.png")