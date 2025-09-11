import pandas as pd
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# ✅ Manually embedded dataset
data = """
CropType,CropDays,SoilMoisture,Temperature,Humidity,Irrigation
Wheat,10,400,30,15,0
Wheat,7,200,30,32,0
Wheat,9,300,21,28,0
Wheat,3,500,40,22,0
Wheat,2,700,23,34,0
Wheat,6,800,21,29,0
Wheat,5,500,33,26,0
Wheat,8,350,21,28,0
Wheat,11,123,17,45,0
Wheat,12,543,25,53,0
Paddy,3,700,35,30,1
Paddy,6,750,32,45,1
Paddy,8,710,34,41,1
Paddy,10,734,38,43,1
Paddy,5,720,30,40,1
Paddy,7,700,33,42,1
Paddy,2,740,36,47,1
Paddy,4,760,37,48,1
Paddy,9,780,35,50,1
Paddy,11,790,39,44,1
Sugarcane,12,350,40,70,1
Sugarcane,13,340,41,72,1
Sugarcane,11,320,39,74,1
Sugarcane,10,310,38,69,1
Sugarcane,9,300,40,71,1
Sugarcane,14,330,41,73,1
Sugarcane,8,315,39,70,1
Sugarcane,15,325,38,75,1
Sugarcane,7,340,37,76,1
Sugarcane,16,345,42,74,1
Barley,3,300,25,23,0
Barley,4,310,28,25,0
Barley,2,290,27,26,0
Barley,5,280,26,24,0
Barley,6,270,25,22,0
Barley,1,260,24,23,0
Barley,7,250,28,27,0
Barley,8,240,29,28,0
Barley,9,230,26,24,0
Barley,10,220,25,26,0
Cotton,6,500,35,40,1
Cotton,7,510,36,42,1
Cotton,5,490,34,41,1
Cotton,4,480,33,43,1
Cotton,3,470,32,39,1
Cotton,8,520,35,44,1
Cotton,2,460,36,45,1
Cotton,9,530,34,46,1
Cotton,10,540,33,40,1
Cotton,1,450,32,38,1
Corn,5,450,31,33,0
Corn,6,460,32,32,0
Corn,4,440,30,31,0
Corn,3,430,29,30,0
Corn,7,470,31,34,0
Corn,2,420,28,29,0
Corn,8,480,32,33,0
Corn,1,410,27,28,0
Corn,9,490,33,32,0
Corn,10,500,34,31,0
Millet,2,300,38,21,0
Millet,3,310,39,22,0
Millet,1,290,37,20,0
Millet,4,320,40,23,0
Millet,5,330,41,24,0
Millet,6,340,42,25,0
Millet,7,350,40,26,0
Millet,8,360,39,27,0
Millet,9,370,41,28,0
Millet,10,380,42,29,0
Tea,20,250,22,90,1
Tea,22,260,23,88,1
Tea,24,240,21,87,1
Tea,26,230,20,89,1
Tea,28,270,22,91,1
Tea,30,280,23,90,1
Tea,32,290,21,88,1
Tea,34,300,24,92,1
Tea,36,310,25,93,1
Tea,38,320,23,91,1
Coffee,90,600,25,20,1
Coffee,92,610,26,22,1
Coffee,94,620,24,19,1
Coffee,96,630,23,18,1
Coffee,98,640,22,21,1
Coffee,100,650,24,20,1
Coffee,102,660,25,19,1
Coffee,104,670,26,22,1
Coffee,106,680,27,23,1
Coffee,108,690,28,24,1
Coffee,99,678,24,18,1
Coffee,101,201,21,14,0
"""

# ✅ Load dataset from the string
df = pd.read_csv(StringIO(data.strip()))

# ✅ Encode crop names to numerical values
df['CropType'] = df['CropType'].astype(str)
crop_map = {name: idx for idx, name in enumerate(df['CropType'].unique())}
df['Crop'] = df['CropType'].map(crop_map)

# ✅ Features and target
X = df[['Crop', 'CropDays', 'SoilMoisture', 'Temperature', 'Humidity']]
y = df['Irrigation']

# ✅ Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ✅ Train the model
print("🚀 Training model...")
rf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
rf.fit(X_train, y_train)

# ✅ Evaluation
y_pred = rf.predict(X_test)
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
print("🧮 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ✅ Save the model and utilities
joblib.dump(rf, "rf_irrigation.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(crop_map, "crop_map.pkl")

print("\n✅ Model, scaler, and crop mapping saved as:")
print("- rf_irrigation.pkl")
print("- scaler.pkl")
print("- crop_map.pkl")


# ✅ Optional: Make predictions using the trained model
def predict_irrigation(crop_name, crop_days, soil_moisture, temperature, humidity):
    model = joblib.load("rf_irrigation.pkl")
    scaler = joblib.load("scaler.pkl")
    crop_map = joblib.load("crop_map.pkl")

    if crop_name not in crop_map:
        raise ValueError(f"❌ Crop '{crop_name}' not found in crop_map.")
    crop_encoded = crop_map[crop_name]

    input_df = pd.DataFrame([{
        'Crop': crop_encoded,
        'CropDays': crop_days,
        'SoilMoisture': soil_moisture,
        'Temperature': temperature,
        'Humidity': humidity
    }])

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    return "💧 Irrigation ON" if prediction == 1 else "⛔ Irrigation OFF"

# ✅ Example prediction
print("\n🔍 Prediction Example:")
print(predict_irrigation("Wheat", 8, 300, 27, 25))

print("\n🔍 Prediction Example:")
print(predict_irrigation("Coffee", 8, 300, 27, 25))

print("\n🔍 Prediction Example:")
print(predict_irrigation("Coffee", 99, 678, 24, 18))

import streamlit as st
import joblib
import pandas as pd

# Load artifacts
model = joblib.load("rf_irrigation.pkl")
scaler = joblib.load("scaler.pkl")
crop_map = joblib.load("crop_map.pkl")

st.title("🌱 Smart Irrigation Predictor")

crop_name = st.selectbox("Select Crop", list(crop_map.keys()))
crop_days = st.number_input("Crop Days", min_value=1, max_value=200, value=10)
soil_moisture = st.number_input("Soil Moisture", min_value=100, max_value=1000, value=400)
temperature = st.number_input("Temperature (°C)", min_value=0, max_value=50, value=25)
humidity = st.number_input("Humidity (%)", min_value=0, max_value=100, value=40)

if st.button("Predict Irrigation"):
    crop_encoded = crop_map[crop_name]
    input_df = pd.DataFrame([{
        "Crop": crop_encoded,
        "CropDays": crop_days,
        "SoilMoisture": soil_moisture,
        "Temperature": temperature,
        "Humidity": humidity
    }])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    if prediction == 1:
        st.success("💧 Irrigation ON")
    else:
        st.error("⛔ Irrigation OFF")