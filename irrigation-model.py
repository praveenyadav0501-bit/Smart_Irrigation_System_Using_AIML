import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Example dataset (replace with your data)
df = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 5],
    'feature2': [10, 20, 30, 40, 50],
    'target': [0, 1, 0, 1, 0]
})

X = df[['feature1', 'feature2']]
y = df['target']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Random Forest model
rf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
rf.fit(X_train, y_train)

# Evaluate
y_pred = rf.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model and scaler
joblib.dump(rf, "rf_irrigation.pkl")
joblib.dump(scaler, "scaler.pkl")
print("✅ Model and scaler saved successfully!")


import joblib
import pandas as pd

# Load saved model and scaler
model = joblib.load("rf_irrigation.pkl")
scaler = joblib.load("scaler.pkl")

def predict_irrigation(crop, soil_moisture, temperature, humidity, rainfall):
    # Construct feature row
    df = pd.DataFrame([{
        "Crop": crop,
        "Soil_Moisture": soil_moisture,
        "Temperature": temperature,
        "Humidity": humidity,
        "Rainfall": rainfall
    }])

    # Encode categorical feature (simple mapping, must match training order!)
    crop_map = {"Wheat": 0, "Rice": 1, "Maize": 2}  # adjust based on dataset
    df["Crop"] = df["Crop"].map(crop_map)

    # Scale
    X = scaler.transform(df)

    # Predict
    pred = model.predict(X)[0]
    return "💧 Irrigation ON" if pred == 1 else "⛔ Irrigation OFF"
