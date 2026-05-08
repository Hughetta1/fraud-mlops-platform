# Deployment Guide

## Option A: Docker Compose (local server)

```bash
# Build and start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f fraud_api
docker-compose logs -f dashboard

# Scale API horizontally
docker-compose up -d --scale fraud_api=3

# Stop
docker-compose down
```

Services:
- API Docs: http://localhost:8000/docs
- Dashboard: http://localhost:8501
- Health: http://localhost:8000/health
- Redis: localhost:6379

---

## Option B: Cloud Deployment (free tier)

### API on Render.com

1. Create account at https://render.com
2. New Web Service → connect your GitHub repo
3. Settings:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn src.api.fraud_api:app --host 0.0.0.0 --port $PORT`
   - Python Version: 3.9+
4. Environment variables:
   - `PYTHONPATH=/opt/render/project/src`
5. Deploy → get URL like `https://fraud-api.onrender.com`

### Dashboard on Streamlit Cloud

1. Create account at https://streamlit.io/cloud
2. New app → connect GitHub repo
3. Main file path: `src/monitoring/dashboard.py`
4. Set `API_BASE_URL` to your Render URL
5. Deploy → get URL like `https://fraud-dashboard.streamlit.app`

### Redis on Redis Cloud

1. Create account at https://redis.com/try-free
2. Create free database (30MB)
3. Set `REDIS_URL` in Render environment variables

---

## Option C: Single VM (AWS EC2 / GCP Compute Engine)

```bash
# Install Docker
curl -fsSL https://get.docker.com | sh

# Clone repo
git clone <your-repo-url>
cd fraud-detection-system

# Start
docker-compose -f docker-compose.yml up -d

# Enable firewall
sudo ufw allow 8000
sudo ufw allow 8501
```

---

## Verifying Deployment

```bash
# Health check
curl https://your-domain.com/health

# Test prediction
curl -X POST https://your-domain.com/predict \
  -H "Content-Type: application/json" \
  -d '{"Time":50000,"Amount":120.50,"V1":-1.35,"V2":-0.07,"V3":2.53,"V4":1.37,"V5":-0.33,"V6":0.46,"V7":0.23,"V8":0.09,"V9":0.36,"V10":0.09,"V11":-0.55,"V12":-0.61,"V13":-0.99,"V14":-0.31,"V15":1.46,"V16":-0.47,"V17":0.20,"V18":0.02,"V19":0.40,"V20":0.25,"transaction_id":"TEST_001"}'

# Check model versions
curl https://your-domain.com/monitoring/versions
```
