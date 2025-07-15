import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score,confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load your dataset
df = pd.read_csv("valid_career_dataset.csv")

# 2. Preprocess skill and personality features
def split_column(column):
    return column.str.split(", ").apply(lambda x: [i.strip() for i in x])

df["skills"] = split_column(df["selected_skills"])
df["personalities"] = split_column(df["personality_types"])

# Use MultiLabelBinarizer for multi-hot encoding
mlb_skills = MultiLabelBinarizer()
mlb_personality = MultiLabelBinarizer()

X_skills = mlb_skills.fit_transform(df["skills"])
X_personality = mlb_personality.fit_transform(df["personalities"])

# Combine encoded inputs
import numpy as np
X = np.hstack((X_skills, X_personality))

# Encode target career labels
le = LabelEncoder()
y = le.fit_transform(df["career_match"])

# 3. Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train XGBoost classifier
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", max_depth=6)
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
print("\n📊 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix - Career Prediction")
plt.xlabel("Predicted Career")
plt.ylabel("Actual Career")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# 6. Save model and encoders
joblib.dump(model, "xgboost_career_model.pkl")
joblib.dump(mlb_skills, "mlb_skills.pkl")
joblib.dump(mlb_personality, "mlb_personality.pkl")
joblib.dump(le, "label_encoder.pkl")

print("\n✅ Model and encoders saved!")
