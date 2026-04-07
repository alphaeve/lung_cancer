import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from xgboost import XGBClassifier
import shap
import joblib
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("survey_lung_cancer.csv")

# Convert YES/NO → 1/0
data.replace({"YES": 1, "NO": 0}, inplace=True)

# Convert Gender
data["GENDER"] = data["GENDER"].replace({"M": 1, "F": 0})

# FIX: Convert 1/2 → 0/1 properly
binary_cols = data.columns.drop(["GENDER", "AGE", "LUNG_CANCER"])
for col in binary_cols:
    data[col] = data[col].replace({1: 0, 2: 1})

# ✅ ADD POLLUTION DATA
data["PM25"] = np.random.randint(40, 120, size=len(data))

# Features & target
X = data.drop("LUNG_CANCER", axis=1)
y = data["LUNG_CANCER"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

# Predictions
pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]

# ✅ Better Evaluation
print("Accuracy:", accuracy_score(y_test, pred))
print("ROC-AUC:", roc_auc_score(y_test, prob))
print(classification_report(y_test, pred))

# Save model
joblib.dump(model, "model.pkl")

# SHAP Explainability
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, show=False)
plt.savefig("shap_feature_importance.png")


# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from xgboost import XGBClassifier
# import shap
# import joblib
# import matplotlib.pyplot as plt

# # Load dataset
# data = pd.read_csv("survey_lung_cancer.csv")

# # Convert YES/NO values to 1/0
# data.replace({"YES": 1, "NO": 0}, inplace=True)

# # Convert GENDER column to numbers
# data["GENDER"] = data["GENDER"].replace({"M": 1, "F": 0})

# # Features and target
# X = data.drop("LUNG_CANCER", axis=1)
# y = data["LUNG_CANCER"]
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE   
import shap
import joblib
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("survey_lung_cancer.csv")

# Convert YES/NO → 1/0
data.replace({"YES": 1, "NO": 0}, inplace=True)

# Convert Gender
data["GENDER"] = data["GENDER"].replace({"M": 1, "F": 0})

# Convert 1/2 → 0/1
binary_cols = data.columns.drop(["GENDER", "AGE", "LUNG_CANCER"])
for col in binary_cols:
    data[col] = data[col].replace({1: 0, 2: 1})

# Add Pollution Data
data["PM25"] = np.random.randint(40, 120, size=len(data))

# Features & target
X = data.drop("LUNG_CANCER", axis=1)
y = data["LUNG_CANCER"]

# ✅ APPLY SMOTE (IMPORTANT)
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

# Train-test split (with stratify for safety)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Model (removed useless warning param)
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

# Predictions
pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]

# Evaluation
print("Accuracy:", accuracy_score(y_test, pred))
print("ROC-AUC:", roc_auc_score(y_test, prob))
print(classification_report(y_test, pred))

# Save model
joblib.dump(model, "model.pkl")

# SHAP Explainability
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, show=False)
plt.savefig("shap_feature_importance.png")
# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # Train model using :contentReference[oaicite:0]{index=0}
# model = XGBClassifier(
#     n_estimators=250,
#     max_depth=6,
#     learning_rate=0.1
# )

# model.fit(X_train, y_train)

# # Prediction
# pred = model.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, pred))

# # Save model
# joblib.dump(model, "model.pkl")

# # Explainable AI using :contentReference[oaicite:1]{index=1}
# explainer = shap.TreeExplainer(model)
# shap_values = explainer.shap_values(X_test)

# shap.summary_plot(shap_values, X_test, show=False)
# plt.savefig("shap_feature_importance.png")