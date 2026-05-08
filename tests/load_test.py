"""
Locust load test for the Fraud Detection API.

Usage:
    # Start the API server first:
    uvicorn src.api.fraud_api:app --host 0.0.0.0 --port 8000

    # Run load test:
    locust -f tests/load_test.py --host=http://localhost:8000

    # Then open http://localhost:8089 in browser to start the test.

    # Headless mode (CLI only):
    locust -f tests/load_test.py --host=http://localhost:8000 \\
        --users 50 --spawn-rate 5 --run-time 60s --headless
"""

import random

import numpy as np
from locust import HttpUser, between, task


class FraudAPILoadTest(HttpUser):
    """
    Simulates realistic traffic to the fraud detection API.

    Tests three endpoints:
    - /health (lightweight, frequent)
    - /predict (single transaction, core workload)
    - /predict/batch (5-20 transactions, batch scenario)
    """

    wait_time = between(0.1, 0.5)  # 2-10 req/s per user

    def on_start(self):
        """Check API health before starting."""
        self.client.get("/health")

    def _random_transaction(self, fraud_prob: float = 0.0017):
        """Generate a realistic synthetic transaction."""
        # Mix of normal and anomalous V-features
        if random.random() < fraud_prob:
            # Fraud-like pattern
            v = {f"V{i}": round(random.gauss(-2, 3), 6) for i in range(1, 29)}
            amount = round(random.lognormal(4.5, 1.2), 2)
        else:
            # Normal pattern
            v = {f"V{i}": round(random.gauss(0, 1), 6) for i in range(1, 29)}
            amount = round(random.lognormal(3.2, 1.0), 2)

        return {
            "Time": float(random.randint(0, 172792)),
            "Amount": max(0.01, amount),
            **v,
            "transaction_id": f"LOAD_{random.randint(100000, 999999)}",
        }

    @task(3)
    def predict_single(self):
        """Single transaction prediction (weight=3, most common)."""
        tx = self._random_transaction()
        with self.client.post("/predict", json=tx, catch_response=True) as resp:
            if resp.status_code == 200:
                data = resp.json()
                prob = data.get("fraud_probability", -1)
                if prob < 0 or prob > 1:
                    resp.failure(f"Invalid probability: {prob}")
            elif resp.status_code == 503:
                resp.failure("Model not loaded")
            else:
                resp.failure(f"HTTP {resp.status_code}")

    @task(1)
    def predict_batch(self):
        """Batch prediction (weight=1, less frequent but heavier)."""
        n = random.randint(3, 15)
        batch = {"transactions": [self._random_transaction() for _ in range(n)]}
        with self.client.post("/predict/batch", json=batch, catch_response=True) as resp:
            if resp.status_code == 200:
                data = resp.json()
                if len(data.get("predictions", [])) != n:
                    resp.failure("Batch count mismatch")
            elif resp.status_code == 503:
                resp.failure("Model not loaded")
            else:
                resp.failure(f"HTTP {resp.status_code}")

    @task(1)
    def health_check(self):
        """Health check (weight=1, lightweight)."""
        self.client.get("/health")

    @task(1)
    def metrics_check(self):
        """Metrics endpoint (weight=1)."""
        self.client.get("/metrics")
