import requests
import pandas as pd
import joblib

URL = "http://localhost:5001/predict"
CSV_PATH = "test_data.csv"

# Load preprocessing tools
label_encoders = joblib.load("label_encoders.pkl")
preprocessor = joblib.load("preprocessor.pkl")

df = pd.read_csv(CSV_PATH)

feature_columns = [
    "Gender","Age","Height","Weight","family_history","FAVC",
    "FCVC","NCP","CAEC","SMOKE","CH2O","SCC","FAF",
    "TUE","CALC","MTRANS"
]

# Tentukan kolom label (Obesity / label)
label_column = "Obesity" if "Obesity" in df.columns else "label"
y = df[label_column].tolist()

# Print mapping class Obesity
print("\n=== Mapping Label Encoder (Obesity) ===")
classes = label_encoders["Obesity"].classes_
for idx, label in enumerate(classes):
    print(f"{label} → {idx}")
print("=======================================\n")

# Copy X agar tidak SettingWithCopyWarning
X = df[feature_columns].copy()

# Apply Label Encoders
for col in label_encoders:
    if col in X.columns:
        X.loc[:, col] = label_encoders[col].transform(X[col])

# Apply Scaler / Preprocessor
numerical_cols = X.select_dtypes(include=['number']).columns
X[numerical_cols] = preprocessor.transform(X[numerical_cols])

# Convert to list
X_list = X.values.tolist()

print(f"Total data untuk inferensi: {len(X_list)}")

# Send prediction request
for i in range(len(X_list)):
    payload = {
        "data": [X_list[i]],
        "label": [y[i]]
    }

    response = requests.post(URL, json=payload)

    if response.status_code == 200:
        result = response.json()
        print(f"\n=== Row {i+1} ===")
        print(f"Input: {X_list[i]}")
        print(f"True Label: {y[i]}")
        print(f"Prediction: {result['predictions']}")
        print(f"Latency: {result['latency']}")
    else:
        print(f"\nError row {i+1}: {response.text}")
