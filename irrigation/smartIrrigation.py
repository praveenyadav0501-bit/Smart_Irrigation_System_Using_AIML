# smartIrrigation.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# ----------------------------
# 1. Load Dataset
# ----------------------------

# Change path for your computer
df = pd.read_csv('CropIrrigationScheduling.csv')

print("Initial Data Preview:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nMissing values per column:\n", df.isnull().sum())

# ----------------------------
# 2. Encode CropType (No mapping Irrigation!)
# ----------------------------

if df['CropType'].dtype == 'object':
    df['CropType'] = df['CropType'].astype('category').cat.codes

# ----------------------------
# 3. Features & Target
# ----------------------------

feature_cols = ['CropType', 'CropDays', 'SoilMoisture', 'Temperature', 'Humidity']
X = df[feature_cols]
y = df['Irrigation']

print("\nTarget distribution:\n", y.value_counts())

# ----------------------------
# 4. Train-Test Split
# ----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ----------------------------
# 5. Train Model
# ----------------------------

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ----------------------------
# 6. Evaluate
# ----------------------------

y_pred = model.predict(X_test)
print("\n✅ Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ----------------------------
# 7. Feature Importance
# ----------------------------

importances = model.feature_importances_
features = X.columns

plt.figure(figsize=(8, 5))
sns.barplot(x=importances, y=features, palette="viridis")
plt.title("Feature Importances in Smart Irrigation Model")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.tight_layout()
plt.show()
