# 🚀 Customer Churn Prediction — Production MLOps System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)
![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face-yellow.svg)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-blue.svg)
![License](https://img.shields.io/badge/License-MIT-red.svg)

**End-to-end machine learning system that predicts customer churn and recommends retention strategies**

[Live API Demo](https://souravmondal619-churn-mlops-api.hf.space/docs) • [Report Bug](../../issues) • [Request Feature](../../issues)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#️-system-architecture)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Usage Guide](#-usage-guide)
- [API Documentation](#-api-documentation)
- [CI/CD Pipeline](#-cicd-pipeline)
- [Business Intelligence](#-business-intelligence)
- [Model Details](#-model-details)
- [Deployment](#-deployment)
- [CrewAI Recommendations](#-crewai-recommendation-agent)
- [Security](#-security--best-practices)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 🎯 Overview

This project delivers a **production-ready customer churn prediction system** that combines machine learning, automated pipelines, and business intelligence to help organizations:

✅ **Predict** which customers are likely to churn  
✅ **Understand** the key drivers behind customer attrition  
✅ **Act** with AI-powered retention recommendations  

The system transforms raw customer data into actionable insights through an automated ML pipeline that trains models, optimizes decision thresholds, deploys APIs, and generates business reports—all without manual intervention.

---

## ✨ Key Features

### 🤖 Machine Learning
- **Multi-model comparison**: Logistic Regression, Random Forest, Gradient Boosting, XGBoost
- **Optimized decision threshold**: Uses Youden's J statistic instead of default 0.5
- **Feature engineering**: Geospatial clustering, encoding, scaling
- **Model persistence**: All artifacts version-controlled and centralized

### 🔄 MLOps Automation
- **Fully automated CI/CD**: Training → Evaluation → Deployment on every push
- **Zero-downtime deployment**: Models automatically sync to production API
- **Artifact management**: Centralized storage on Hugging Face Model Hub
- **Batch scoring**: Excel input → CSV predictions with business flags

### 🌐 Production API
- **FastAPI REST service**: Real-time predictions with automatic docs
- **Hugging Face Spaces hosting**: Scalable, serverless deployment
- **Multi-format output**: Probability scores, binary flags, class labels
- **Swagger UI**: Interactive API testing built-in

### 📊 Business Intelligence
- **Power BI dashboard**: Visual analytics on churn patterns
- **Explainable insights**: Contract type, tenure, service usage correlations
- **Actionable segments**: High-risk customer identification

### 🤝 AI-Powered Recommendations
- **CrewAI integration**: Generates personalized retention strategies
- **Context-aware suggestions**: Discounts, loyalty rewards, contract upgrades
- **Decision support**: Transforms predictions into business actions

---

## 🏗️ System Architecture

```
┌─────────────────┐
│   Raw Dataset   │
│ churn_clean.xlsx│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  preprocess.py  │  ← Data cleaning, encoding, geospatial clustering
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    train.py     │  ← Model training + threshold optimization
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   evaluate.py   │  ← Performance metrics & validation
└────────┬────────┘
         │
         ├──────────────────────────┐
         │                          │
         ▼                          ▼
┌──────────────────┐      ┌─────────────────────┐
│ batch_predict.py │      │   FastAPI (api.py)  │
│ Excel → CSV      │      │   + HF Spaces       │
└──────────────────┘      └─────────────────────┘
         │                          │
         ▼                          ▼
┌──────────────────┐      ┌─────────────────────┐
│ GitHub Artifacts │      │  Live REST Endpoint │
└──────────────────┘      └─────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│      GitHub Actions CI/CD           │
│  Auto-train → Test → Deploy → Sync │
└─────────────────────────────────────┘
```

---

## 📁 Project Structure

```
MLOps-churn-prediction-system/
│
├── 📂 data/
│   └── churn_clean.xlsx              # Training dataset
│
├── 📂 models/                        # Auto-generated artifacts (not committed)
│   ├── model.pkl                     # Trained model
│   ├── scaler.pkl                    # Feature scaler
│   ├── label_encoder.pkl             # Target encoder
│   ├── kmeans.pkl                    # Geospatial clustering model
│   ├── feature_columns.pkl           # Feature schema
│   └── churn_threshold.pkl           # Optimized decision boundary
│
├── 📂 new_test_data/
│   ├── new_data.xlsx                 # Business input for batch predictions
│   └── predictions_output.csv        # Generated predictions
│
├── 📂 src/
│   ├── preprocess.py                 # Data preprocessing pipeline
│   ├── train.py                      # Model training + optimization
│   ├── evaluate.py                   # Model evaluation metrics
│   ├── predict.py                    # Single-customer inference
│   ├── batch_predict.py              # Bulk prediction pipeline
│   └── api.py                        # FastAPI service
│
├── 📂 notebooks/
│   └── Teleco_Customer_Churn_Analysis.ipynb  # EDA & experimentation
│
├── 📂 dashboards/
│   └── churn_powerbi.pbix            # Power BI analytics dashboard
│
├── 📂 .github/workflows/
│   └── ci.yml                        # CI/CD automation pipeline
│
├── Dockerfile                        # API containerization
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip package manager
- Git
- (Optional) Docker for containerization

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/MLOps-churn-prediction-system.git
cd MLOps-churn-prediction-system
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Verify data availability**
```bash
# Ensure training data exists
ls data/churn_clean.xlsx
```

### Local Execution

Run the complete pipeline locally:

```bash
# Step 1: Preprocess data
python src/preprocess.py

# Step 2: Train model with threshold optimization
python src/train.py

# Step 3: Evaluate model performance
python src/evaluate.py

# Step 4: Generate batch predictions
python src/batch_predict.py
```

---

## 📖 Usage Guide

### 1️⃣ Updating Training Data

**To retrain with new customer data:**

```bash
# 1. Update the training dataset
cp your_new_data.xlsx data/churn_clean.xlsx

# 2. Commit and push changes
git add data/churn_clean.xlsx
git commit -m "Update training data with Q4 2024 customers"
git push origin main
```

**What happens next:**
- GitHub Actions automatically triggers
- Model retrains with new data
- Artifacts upload to Hugging Face
- Production API updates automatically
- No manual deployment needed ✅

### 2️⃣ Batch Predictions (Excel → CSV)

**For business teams to score customer lists:**

```bash
# 1. Place your customer data file
cp customer_list.xlsx new_test_data/new_data.xlsx

# 2. Run batch prediction
python src/batch_predict.py

# 3. Retrieve results
cat new_test_data/predictions_output.csv
```

**Output format:**
```csv
Customer ID,Prediction,Churn_Flag,Churn_Probability,Threshold_Used
0001-BGFD,Joined,YES,0.48,0.25
0002-XKTF,Stayed,NO,0.12,0.25
```

### 3️⃣ Single Customer Prediction

```bash
python src/predict.py
```

Provide customer details when prompted, or modify the script for programmatic use.

---

## 🌐 API Documentation

### Live Endpoint

**Interactive Swagger Docs**: [https://souravmondal619-churn-mlops-api.hf.space/docs](https://souravmondal619-churn-mlops-api.hf.space/docs)

### Sample Request

**POST** `/predict`

```json
{
  "Customer ID": "0001-BGFD",
  "Monthly Charge": 75,
  "Total Revenue": 2800,
  "Tenure Months": 24,
  "Latitude": 40.7,
  "Longitude": -73.9,
  "Gender": "Male",
  "Senior Citizen": "No",
  "Internet Service": "Fiber Optic",
  "Contract": "Month-to-Month",
  "Payment Method": "Credit Card"
}
```

### Sample Response

```json
{
  "customer_id": "0001-BGFD",
  "prediction": "Joined",
  "churn_flag": "YES",
  "churn_probability": 0.48,
  "threshold_used": 0.25,
  "probabilities": {
    "Churned": 0.48,
    "Joined": 0.51,
    "Stayed": 0.01
  }
}
```

### Python Client Example

```python
import requests

url = "https://souravmondal619-churn-mlops-api.hf.space/predict"
data = {
    "Customer ID": "0001-BGFD",
    "Monthly Charge": 75,
    "Total Revenue": 2800,
    "Tenure Months": 24,
    "Latitude": 40.7,
    "Longitude": -73.9,
    "Gender": "Male",
    "Senior Citizen": "No",
    "Internet Service": "Fiber Optic",
    "Contract": "Month-to-Month",
    "Payment Method": "Credit Card"
}

response = requests.post(url, json=data)
print(response.json())
```

---

## 🔄 Model Retraining, CI/CD & Deployment Flow (Crystal Clear)

This project is designed so that **model training, evaluation, batch prediction, and deployment are automated** as much as possible. Here is the complete picture 👇

### 📥 1️⃣ How to Feed New Data Into the Model

Training data lives in:
```
data/churn_clean.xlsx
```

**To retrain with new data:**

```bash
# 1. Append or replace rows in the training dataset
cp your_new_data.xlsx data/churn_clean.xlsx

# 2. Commit and push your changes
git add data/churn_clean.xlsx
git commit -m "Update training data"
git push origin main
```

🔔 **No manual retraining needed locally** — GitHub Actions handles it automatically!

### 🤖 2️⃣ What Happens After You Push (CI/CD Pipeline)

When you push to `main`, the CI pipeline **automatically runs**:

**Pipeline Stages:**
```yaml
1️⃣ Install dependencies
2️⃣ Preprocessing (data cleaning + feature engineering)
3️⃣ Model Training (with threshold optimization)
4️⃣ Model Evaluation (metrics & validation)
5️⃣ Upload artifacts to Hugging Face Model Hub
6️⃣ Run batch predictions
7️⃣ Build Docker API image
8️⃣ Upload prediction results as GitHub artifacts
```

**Pipeline file location:**
```
.github/workflows/ci.yml
```

### 🚀 3️⃣ Does Hugging Face Model Update Automatically?

✅ **YES — completely automatically!**

When training finishes, CI uploads artifacts directly to the HF Model Hub.

**Files updated:**
```
model.pkl
scaler.pkl
kmeans.pkl
label_encoder.pkl
feature_columns.pkl
churn_threshold.pkl
```

**Repository location:**
```
hf: souravmondal619/churn-mlops-model
```

**What this means:**
- When the model retrains → Hugging Face model is updated automatically
- Since the API loads artifacts directly from Hugging Face → **The deployed API always uses the latest trained model**

### 🧩 4️⃣ Does the API Deployment Update Too?

**Yes — indirectly and automatically!**

The API (`api.py`) loads models **dynamically** from Hugging Face:

```python
from huggingface_hub import hf_hub_download

model = pickle.load(open(hf_hub_download(REPO_ID, "model.pkl"), "rb"))
```

**So after CI uploads a new model:**
- ➡️ Hugging Face API Space automatically starts using the new version
- ➡️ **No manual re-deployment needed**

### 🏗 5️⃣ What About the CI Pipeline Model?

The CI pipeline uses the same code, so it:
- ✔ Retrains the model
- ✔ Evaluates it
- ✔ Uploads it to Hugging Face
- ✔ Generates prediction artifacts

**Everything stays synchronized.**  
There is only **ONE source of truth** — the Hugging Face model repository.

### 🔐 6️⃣ Connecting GitHub → Hugging Face (via Token)

To let GitHub upload artifacts securely, we use a Hugging Face Access Token.

**Step 1 — Create Token in Hugging Face**

Go to: [Settings → Access Tokens → New Token](https://huggingface.co/settings/tokens)

**Permissions required:**
- ✔ Write access to repositories
- ✔ Read access (auto)
- Name it anything (e.g., `HF_TOKEN`)

Copy the generated token.

**Step 2 — Store Token in GitHub Secrets**

Navigate to: `GitHub → Repository → Settings → Secrets → Actions → New repository secret`

```
Name: HF_TOKEN
Value: [Your Hugging Face token - keep this private!]
```

**Step 3 — CI Uses the Token Securely**

Inside `ci.yml`, we authenticate with Hugging Face:

```yaml
- name: Login to Hugging Face
  run: |
    python -c "from huggingface_hub import login; import os; login(token=os.environ['HF_TOKEN'])"
  env:
    HF_TOKEN: ${{ secrets.HF_TOKEN }}
```

**Now:**
- ✔ CI can upload models securely
- ✔ No one sees your token
- ✔ Fully automated and secure

### 📬 7️⃣ Feeding JSON to API (For Inference)

**Swagger documentation:**
```
https://souravmondal619-churn-mlops-api.hf.space/docs
```

**Example request body:**
```json
{
  "Customer ID": "0001-BGFD",
  "Monthly Charge": 75,
  "Total Revenue": 2800,
  "Tenure Months": 24,
  "Latitude": 40.7,
  "Longitude": -73.9,
  "Gender": "Male",
  "Senior Citizen": "No",
  "Internet Service": "Fiber Optic",
  "Contract": "Month-to-Month",
  "Payment Method": "Credit Card"
}
```

**Example API response:**
```json
{
  "prediction": "Joined",
  "churn_flag": "YES",
  "churn_probability": 0.49,
  "threshold_used": 0.25
}
```

### 🧾 Summary (Bullet-Proof Clarity)

| Action | Result |
|--------|--------|
| Modify training data | Model retrains automatically |
| Push to GitHub | CI pipeline runs |
| Pipeline finishes | New model uploaded to HF |
| API calls | Always use latest model |
| GitHub Artifacts | Store batch prediction CSVs |
| HF Token | Secure bridge between GitHub → HuggingFace |

**Everything is automated. No manual deployments. No duplicated models.**

---

## 📊 Business Intelligence

### Power BI Dashboard Insights

**Key findings from exploratory analysis:**

📌 **Contract type is the strongest churn indicator**
- Month-to-month contracts show highest churn rates
- Two-year contracts have 90% lower churn

⚠️ **High-risk customer segments**
- Tenure < 12 months: 45% churn rate
- Total charges > $5000 + complaints: 67% churn rate
- Fiber optic users with month-to-month contracts: 52% churn

🔌 **Service-level patterns**
- Fiber optic internet: Higher churn (price sensitivity)
- Credit card auto-pay: 30% lower churn
- Senior citizens: 1.4x higher churn rate

💡 **Retention opportunities**
- Contract upgrades reduce churn by 65%
- Loyalty rewards effective for high-value customers
- Proactive support for first-year customers critical

### Dashboard Components

- Churn rate by contract type
- Tenure vs churn probability
- Revenue distribution across churn segments
- Service usage patterns
- Geographic churn clusters
- Payment method impact analysis

### EDA Findings

**From notebooks analysis:**

- **Revenue vs churn is non-linear**: High-value customers churn at different rates
- **Internet service type plays a major role**: Fiber optic shows unique patterns
- **Senior citizens churn at a higher rate**: 1.4x baseline rate
- **Missing values handling**: Systematic imputation strategies applied
- **Outlier detection**: Charges, downloads, revenue carefully examined
- **Correlation heatmaps**: Strong correlations between contract/tenure/churn

---

## 🤖 Model Details

### Data Preprocessing

**Feature Engineering:**
- Geospatial clustering (latitude/longitude → cluster IDs)
- Categorical encoding (one-hot + label encoding)
- Numerical scaling (StandardScaler)
- Missing value imputation
- Outlier detection and handling

**Transformations:**
- Revenue binning
- Tenure categorization
- Interaction features (contract × service type)

### Model Training

**Algorithms Evaluated:**
- Logistic Regression (baseline)
- Random Forest
- Gradient Boosting
- XGBoost

**Selection Criteria:**
- Cross-validated accuracy
- Recall on churn class (minimize false negatives)
- Model interpretability
- Inference latency

**Final Model**: Gradient Boosting (best balance of performance and explainability)

### Threshold Optimization

Instead of using the default 0.5 probability threshold, we optimize using:

1. **ROC Curve Analysis**: Find operating point balancing TPR/FPR
2. **Youden's J Statistic**: Maximize (Sensitivity + Specificity - 1)
3. **Business Cost Sensitivity**: Account for retention cost vs churn cost

**Result**: Optimized threshold (~0.25) saved as `churn_threshold.pkl`

**Benefits:**
- Consistent predictions across all environments
- Better alignment with business objectives
- Reduced false negatives (missed churners)

### Model Performance

**Validation Metrics** (typical results):
- Accuracy: 82%
- Precision (Churn): 74%
- Recall (Churn): 81%
- F1-Score: 77%
- ROC-AUC: 0.87

---

## 🚀 Deployment

### Hugging Face Spaces

**Why Hugging Face?**
- Serverless deployment (zero infrastructure management)
- Automatic scaling
- Built-in monitoring
- Free tier available
- Direct integration with Model Hub

**Deployment Process:**

1. **CI uploads artifacts** → Hugging Face Model Hub
2. **FastAPI loads artifacts** dynamically at runtime
3. **Spaces hosts API** with automatic SSL/DNS
4. **Users access** via public endpoint


### Docker Deployment

**Why Docker?**

This project uses a **Dockerfile** so the API runs the **same way everywhere** — locally, on Hugging Face Spaces, or on any cloud — **without breaking**.

**Benefits:**
- ✅ **Environment consistency**: Same dependencies, same Python version, same behavior
- ✅ **Reproducibility**: No "works on my machine" issues
- ✅ **Portability**: Deploy anywhere that supports Docker (AWS, GCP, Azure, HF Spaces)
- ✅ **Isolation**: Clean, containerized environment

**Build and run locally:**

```bash
# Build image
docker build -t churn-api .

# Run container
docker run -p 8000:8000 churn-api

# Access API
curl http://localhost:8000/docs
```

**Hugging Face Spaces uses the same Dockerfile**, ensuring identical behavior between your local development and production deployment.

### Automatic Synchronization Flow

```
┌──────────────────────────────────────────────────────────────┐
│ 1. Developer pushes code/data                                │
└────────────────┬─────────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────────┐
│ 2. GitHub Actions triggers CI pipeline                       │
└────────────────┬─────────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────────┐
│ 3. Model retrains with new data                              │
└────────────────┬─────────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────────┐
│ 4. Artifacts upload to Hugging Face Model Hub                │
└────────────────┬─────────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────────┐
│ 5. Production API automatically uses new model               │
│    (no restart required)                                     │
└──────────────────────────────────────────────────────────────┘
```

**Summary Table:**

| Action | Result |
|--------|--------|
| Modify training data | Model retrains automatically |
| Push to GitHub | CI pipeline runs |
| Pipeline finishes | New model uploaded to HF |
| API calls | Always use latest model |
| GitHub Artifacts | Store batch prediction CSVs |
| HF Token | Secure bridge between GitHub → HuggingFace |

**No manual deployment needed. Ever.**

---

## 🤝 CrewAI Recommendation Agent Implementation In Notebook 

### Purpose

Transform predictions into actionable retention strategies using AI-powered analysis.

### How It Works

1. **Input**: Customer profile + churn probability
2. **Analysis**: CrewAI agent evaluates risk factors
3. **Output**: Personalized retention recommendations

### Sample Recommendations

**For high-risk month-to-month customer:**
- Offer 20% discount on annual contract upgrade
- Enroll in loyalty rewards program
- Schedule proactive support call
- Provide fiber optic speed boost trial

**For moderate-risk customer:**
- Highlight contract upgrade benefits
- Offer flexible payment options
- Send personalized retention email

### Integration Example
```
strategy_agent = Agent(
    role="Telecom Retention Strategist",
    goal="Provide retention recommendations based on churn risk",
    backstory="You are a senior telecom business strategist...",
    llm=llm,
    verbose=False
)

for _, r in results.iterrows():
    if risk in ["Medium Risk", "High Risk"]:
        task = Task(
            description=f"Customer churn probability: {churn_prob:.2f}, Risk: {risk}",
            expected_output="One concise retention recommendation.",
            agent=strategy_agent
        )
        recommendation = Crew(agents=[strategy_agent], tasks=[task]).kickoff()
```

---

## 📧 Contact

**Project Maintainer**: Sourav Mondal  

---