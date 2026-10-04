FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy exporter file
COPY prometheus_exporter.py /app/prometheus_exporter.py

# Install dependencies
RUN pip install --no-cache-dir \
    prometheus_client \
    flask \
    psutil \
    requests \
    scikit-learn

# Expose ports:
# 8000 = Prometheus metrics
# 5001 = Endpoint /predict (internal)
EXPOSE 8000
EXPOSE 5001

CMD ["python", "prometheus_exporter.py"]
