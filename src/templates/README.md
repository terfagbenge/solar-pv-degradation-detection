# ☀️ Solar PV Degradation Detection System

An end-to-end Machine Learning web application that detects progressive degradation in Solar Photovoltaic (PV) strings. Trained on the **PV-Fault-DS226** dataset, this system uses a Random Forest classifier to identify degradation and evaluates the system using **Detection Delay** and **False Alarm Rate (FAR)**.

## 🛠️ Tech Stack
* **Machine Learning:** Scikit-Learn (Random Forest), Pandas, NumPy
* **Backend:** FastAPI, Uvicorn, Pydantic
* **Frontend:** HTML, CSS, JavaScript (Chart.js)
* **Dataset:** Zenodo PV-Fault-DS226 (Time-series electrical measurements)

## 📊 Model Performance
* **String 1:** 99.93% Accuracy | 0.04% False Alarm Rate | 3-Row Persistence Check
* **String 2:** 99.79% Accuracy | 0.08% False Alarm Rate | 3-Row Persistence Check

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/solar-pv-degradation-detection.git
   cd solar-pv-degradation-detection