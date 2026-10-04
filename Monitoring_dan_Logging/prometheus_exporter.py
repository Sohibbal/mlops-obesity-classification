from prometheus_client import start_http_server, Counter, Gauge
from flask import Flask, request, jsonify
import psutil
import time
import requests
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# =========================
# PROMETHEUS METRICS
# =========================
REQUEST_COUNT = Counter("request_count", "Total inference request diterima exporter")
REQUEST_LATENCY_AVG = Gauge("request_latency_avg_seconds", "Rata-rata latency inferensi detik")
CPU_USAGE = Gauge("cpu_usage_percent", "CPU usage exporter host (%)")
RAM_USAGE = Gauge("ram_usage_percent", "RAM usage exporter host (%)")
DISK_USAGE = Gauge("disk_usage_percent", "Disk usage root (%)")
SLA_BREACH = Counter("sla_breach_count", "Jumlah request lambat > 1 detik")
ERROR_COUNT = Counter("error_count", "Total error selama inferensi")
PAYLOAD_SIZE = Gauge("payload_size_bytes", "Ukuran payload request terakhir (bytes)")

MODEL_ACCURACY = Gauge("model_accuracy", "Akurasi runtime model")
MODEL_F1 = Gauge("model_f1_score", "F1-score runtime model")
MODEL_PRECISION = Gauge("model_precision", "Precision runtime model")
MODEL_RECALL = Gauge("model_recall", "Recall runtime model")

# Runtime tracking metrics
CORRECT_PRED = Counter("correct_predictions", "Total prediksi benar")
TOTAL_PRED = Counter("total_predictions", "Total prediksi dengan label")

# Store lists for cumulative metrics
y_true_history = []
y_pred_history = []

# Latency tracking
total_latency = 0.0
latency_count = 0

# =========================
# FLASK APP
# =========================
app = Flask(__name__)
MLFLOW_MODEL_URL = "http://model-service:8080/invocations"


def update_system_metrics():
    CPU_USAGE.set(psutil.cpu_percent(interval=None))
    RAM_USAGE.set(psutil.virtual_memory().percent)
    DISK_USAGE.set(psutil.disk_usage("/").percent)


@app.route("/predict", methods=["POST"])
def predict():
    global total_latency, latency_count

    try:
        input_json = request.get_json()
        payload_bytes = len(request.data)
        PAYLOAD_SIZE.set(payload_bytes)

        REQUEST_COUNT.inc()
        start_time = time.time()

        model_response = requests.post(
            MLFLOW_MODEL_URL,
            json={
                "dataframe_split": {
                    "columns": [
                        "Gender", "Age", "Height", "Weight", "family_history", "FAVC",
                        "FCVC", "NCP", "CAEC", "SMOKE", "CH2O", "SCC", "FAF",
                        "TUE", "CALC", "MTRANS"
                    ],
                    "data": input_json["data"]
                }
            },
            timeout=10
        )

        latency = time.time() - start_time

        # Update average latency
        total_latency += latency
        latency_count += 1
        avg_latency = total_latency / latency_count
        REQUEST_LATENCY_AVG.set(avg_latency)

        if latency > 1:
            SLA_BREACH.inc()

        if model_response.status_code != 200:
            ERROR_COUNT.inc()
            return jsonify({"error": "Model service error", "details": model_response.text}), 500

        predictions = model_response.json()["predictions"]

        # Runtime evaluation if labels provided
        if "label" in input_json:
            y_true = input_json["label"]
            y_pred = predictions

            for t, p in zip(y_true, y_pred):
                TOTAL_PRED.inc()
                y_true_history.append(t)
                y_pred_history.append(p)
                if t == p:
                    CORRECT_PRED.inc()

            if len(set(y_true_history)) > 1:
                MODEL_ACCURACY.set(accuracy_score(y_true_history, y_pred_history))
                MODEL_PRECISION.set(precision_score(y_true_history, y_pred_history, average="macro"))
                MODEL_RECALL.set(recall_score(y_true_history, y_pred_history, average="macro"))
                MODEL_F1.set(f1_score(y_true_history, y_pred_history, average="macro"))

        update_system_metrics()

        return jsonify({
            "predictions": predictions,
            "latency": latency,
            "payload_bytes": payload_bytes
        })

    except Exception as e:
        ERROR_COUNT.inc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("🚀 Prometheus Exporter berjalan di port 8000 ...")
    start_http_server(8000)
    app.run(host="0.0.0.0", port=5001)
