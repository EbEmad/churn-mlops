# Customer Churn Prediction System

This project is a machine learning solution for predicting customer churn in a telecom company. It includes preprocessing, model training, and serving via a FastAPI API. The system is containerized using Docker and orchestrated with Docker Compose.

---

## 📁 Project Structure

```bash
MLOPS_ZOOMCAMP_Project/
│
├── Data/                   # Raw dataset
│   └── churn-data.csv
├── models/                 # Trained model and preprocessing artifacts
│   ├── best_model.pkl
│   ├── columns.pkl
│   └── scaler.pkl
├── notebooks/              # Jupyter notebooks for testing
│   └── test.ipynb
├── scripts/                # Source code for API and testing
│   ├── api.py              # FastAPI inference API
│   └── api_test.ipynb      # Notebook to test API
│
├── Dockerfile              # Docker image definition
├── docker-compose.yml      # Docker Compose setup
├── requirements.txt        # Python dependencies
├── run.sh                  # Script to run the project
├── LICENSE                 # Project license
└── README.md               # This file
```

---

## 🔄 Pipeline Overview

1. **Preprocessing & Model Training**

   * Performed offline; artifacts saved in `models/`.
   * Artifacts include trained model (`best_model.pkl`), feature columns (`columns.pkl`), and scaler (`scaler.pkl`).

2. **FastAPI Inference Server** (`scripts/api.py`)

   * Loads the trained model and artifacts.
   * Accepts JSON input and returns churn predictions.

3. **Testing & Interaction**

   * `api_test.ipynb` demonstrates how to test the API via code.
   * `test.ipynb` allows exploration and validation.

---

## 🛠️ Setup Instructions

### 1. Clone and Install Dependencies

```bash
git clone https://github.com/EbEmad/MLOPS_ZOOMCAMP_Project.git
cd MLOPS_ZOOMCAMP_Project
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run API Server

```bash
uvicorn scripts.api:app --reload
```

### 3. Sample API Request

Use this JSON as input (`sample_input.json`):

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No phone service",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "Yes",
  "StreamingTV": "No",
  "StreamingMovies": "Yes",
  "Contract": "One year",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 70.5,
  "TotalCharges": "845.5"
}
```

Then run:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample_input.json
```

---

## 🐳 Run with Docker

To build and serve using Docker:

```bash
docker-compose up --build
```

---

## ✅ Future Improvements

* Add model training pipeline inside container
* CI/CD integration (e.g., GitHub Actions)
* Cloud deployment (e.g., AWS/GCP)
* Monitoring and logging (e.g., Prometheus + Grafana)
* Model retraining automation

---

## 👤 Author

* **Ebrahim Emad**

