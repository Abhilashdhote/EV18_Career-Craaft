🔧 Physics-Aware ICS/SCADA Anomaly Detection System
📌 Overview

This project implements a physics-aware, explainable anomaly detection system for Industrial Control Systems (ICS/SCADA).
It passively monitors sensor data from a thermal power plant environment and detects rare operational, physical, or cyber anomalies without disrupting production.

The solution combines Autoencoders, Isolation Forest, and XGBoost, along with SHAP-based explainability, and is deployed using an interactive Streamlit dashboard.

🏭 Problem Context

Industrial Control Systems operate critical infrastructure and require:

Near-zero downtime

Safety-compliant monitoring

Explainable alerts for operators

Traditional cybersecurity solutions are unsuitable due to active intervention and lack of physical awareness. This system addresses these challenges using passive, physics-informed machine learning.

🧠 Solution Architecture

Data Flow Pipeline:

ICS sensor data ingestion (passive)

Data preprocessing & normalization

Physics-based feature engineering

Autoencoder anomaly scoring

Isolation Forest structural anomaly detection

XGBoost final anomaly classification

SHAP explainability for detected anomalies

Streamlit dashboard visualization

📁 Project Structure
📦 Round 3
├── app.py                         # Streamlit application (inference + SHAP)
├── enhanced_training_with_shap.py # Model fine-tuning & SHAP analysis
├── ML-Model.ipynb                 # Main training notebook
├── ics_autoencoder.h5             # Trained autoencoder model
├── ics_xgboost.pkl                # Trained XGBoost classifier
├── scaler.pkl                     # StandardScaler used during training
├── thermal_power_ics_combined_dataset.xlsx
├── thermal_plant_preprocessed.xlsx
└── README.md

⚙️ Models Used
🔹 Autoencoder

Learns normal operational behavior

High reconstruction error indicates anomaly

Physics-consistent latent representation

🔹 Isolation Forest

Unsupervised anomaly detection

Identifies rare, structurally different patterns

🔹 XGBoost Classifier

Final anomaly classification

Uses raw features, physics-based features, and anomaly scores

Handles non-linear interactions effectively

🔬 Physics-Aware Feature Engineering

The model incorporates physical system knowledge, including:

Energy balance (Power input vs output)

System efficiency metrics

Flow–pressure–temperature consistency

Control mismatches (pump speed vs valve position)

Non-linear interaction features

This ensures robust detection and meaningful explanations.

📊 Model Performance
Metric	Score
Training Accuracy	99%
Testing Accuracy	96%
Anomaly Recall	High
False Positives	Low

Synthetic anomalies were introduced to ensure realistic evaluation.

🧠 Explainability (SHAP)

SHAP is applied to XGBoost predictions

Explains why a data point was flagged as anomalous

Highlights key contributing parameters

Enables operator trust and regulatory compliance

🖥️ Streamlit Dashboard

Features:

Manual sensor data input

Real-time anomaly prediction

Anomaly severity score

SHAP explanation plots

Clean, operator-friendly UI

To run the dashboard:

streamlit run app.py

🔐 Safety & Compliance

Fully passive monitoring

No control signal injection

Safe for live industrial environments

Designed for explainability and auditability

🚀 Future Enhancements

Real-time streaming integration

Edge deployment

Root-cause classification

Digital twin integration

Predictive maintenance extension

🏁 Conclusion

This project delivers a production-ready, explainable, physics-aware anomaly detection framework for ICS/SCADA systems.
It bridges physical system understanding and advanced machine learning, enabling safe, accurate, and trustworthy monitoring of critical infrastructure.
