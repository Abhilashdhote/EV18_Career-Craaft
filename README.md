Physics-Aware ICS/SCADA Anomaly Detection System
<div align="center">
Passive • Physics-Aware • Explainable • Production-Safe
<img src="https://img.shields.io/badge/ICS%2FSCADA-Safe-green" /> <img src="https://img.shields.io/badge/Explainability-SHAP-blue" /> <img src="https://img.shields.io/badge/Accuracy-96%25-brightgreen" /> <img src="https://img.shields.io/badge/Deployment-Streamlit-orange" /> </div>
What This Project Solves

Problem:
Traditional cybersecurity tools interfere with live industrial systems and lack physical awareness.

Solution:
A passive anomaly detection system that understands industrial physics, detects anomalies accurately, and explains every alert.

Key Capabilities
-
<table> <tr> <td><b>Passive Monitoring</b><br>No control signal injection</td> <td><b>Physics Awareness</b><br>Energy & flow consistency</td> <td><b>Explainable AI</b><br>SHAP-based insights</td> <td><b>Production Ready</b><br>Streamlit deployment</td> </tr> </table>
Architecture Overview
ICS Sensors
   ↓
Preprocessing & Scaling
   ↓
Physics-Based Feature Engineering
   ↓
Autoencoder (Reconstruction Error)
   ↓
Isolation Forest (Structural Anomaly)
   ↓
XGBoost (Final Classification)
   ↓
SHAP Explainability
   ↓
Streamlit Operator Dashboard

Project Structure
<details> <summary><b>Click to expand folder structure</b></summary>
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

</details>

Models Used
-
Model	Purpose
Autoencoder	Learn normal operational behavior
Isolation Forest	Detect rare structural anomalies
XGBoost	Final anomaly classification
Physics-Aware Intelligence
This system embeds real-world physical constraints, including:
Energy balance (power input vs output)
Thermal efficiency monitoring
Flow–pressure–temperature consistency
Actuator–sensor mismatch detection
Non-linear physical interactions
These constraints significantly reduce false positives and improve operator trust.

Model Performance
-
<table> <tr><td><b>Training Accuracy</b></td><td>99%</td></tr> <tr><td><b>Testing Accuracy</b></td><td>96%</td></tr> <tr><td><b>Anomaly Recall</b></td><td>High</td></tr> <tr><td><b>False Positives</b></td><td>Low</td></tr> </table>
Synthetic anomalies were injected to ensure realistic evaluation.
Explainability (SHAP)
Each detected anomaly includes:
Feature-level contribution scores
Direction of influence on prediction
Operator-interpretable reasoning
Audit-ready explanations for compliance

Streamlit Dashboard
Features 
-
Manual sensor data input
Real-time anomaly prediction
Anomaly severity scoring
SHAP explanation plots
Operator-friendly UI
Run locally
streamlit run app.py
Safety and Compliance
Fully passive monitoring
No control signal injection
Safe for live industrial environments
Designed for explainability and auditability

Conclusion
-

This project delivers a production-ready, explainable, physics-aware anomaly detection framework for ICS/SCADA systems.
It successfully bridges physical system understanding with advanced machine learning, enabling safe, accurate, and trustworthy monitoring of critical infrastructure.
