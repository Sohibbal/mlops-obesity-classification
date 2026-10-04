import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os


# Set Mlflow Tracking Uri
mlflow.set_tracking_uri("http://127.0.0.1:5000")  # sama seperti MLflow server
mlflow.set_experiment("Obesity-Classification")


# Menggunakan Mlflow AutoLog
mlflow.sklearn.autolog(log_models=True)

# Meload dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "obesity_classification_preprocessing.csv")

df = pd.read_csv(csv_path)

# Inisialisasi kolom Fitur dan target(Obesity)
X = df.drop(df.columns[-1], axis=1)
y = df[df.columns[-1]]

# Spliting data (Traingin 80%, Testing 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# MLflow Tracking (run name)
with mlflow.start_run(run_name="RandomForest-Autolog"):

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print("Akurasi:", acc)

print("Training metrics sudah dicatat menggunakan autolog MLflow!")