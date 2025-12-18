import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os

# Try to import optional dependencies
try:
    from tensorflow import keras
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# Page configuration
st.set_page_config(
    page_title="Enhanced ICS Anomaly Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(120deg, #1f77b4 0%, #ff7f0e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .alert-critical {
        background: linear-gradient(135deg, #ff4444 0%, #cc0000 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        font-weight: bold;
        font-size: 1.3rem;
        text-align: center;
        box-shadow: 0 6px 12px rgba(255,0,0,0.3);
        animation: pulse 2s infinite;
    }
    .alert-warning {
        background: linear-gradient(135deg, #ffbb33 0%, #ff8800 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        font-weight: bold;
        font-size: 1.3rem;
        text-align: center;
        box-shadow: 0 6px 12px rgba(255,136,0,0.3);
    }
    .alert-normal {
        background: linear-gradient(135deg, #00C851 0%, #007E33 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        font-weight: bold;
        font-size: 1.3rem;
        text-align: center;
        box-shadow: 0 6px 12px rgba(0,200,81,0.3);
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }
    .shap-explanation {
        background: #f8f9fa;
        padding: 1rem;
        border-left: 4px solid #1f77b4;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(to right, #00C851, #ffbb33, #ff4444);
    }
</style>
""", unsafe_allow_html=True)

# Check model files
@st.cache_data
def check_model_files():
    """Check if required model files exist"""
    required_files = {
        'ics_xgboost.pkl': 'XGBoost classifier',
        'scaler.pkl': 'Feature scaler'
    }
    
    missing = []
    for file, desc in required_files.items():
        if not os.path.exists(file):
            missing.append(f"❌ {file} ({desc})")
    
    return missing

# Load models with proper error handling
@st.cache_resource
def load_models():
    """Load all trained models and components"""
    missing = check_model_files()
    
    if missing:
        return None, None, None, None, None
    
    try:
        xgb_model = joblib.load('ics_xgboost.pkl')
        scaler = joblib.load('scaler.pkl')
        
        # Load feature names
        try:
            feature_names = joblib.load('feature_names.pkl')
        except:
            feature_names = None
        
        # Try to load autoencoder
        autoencoder = None
        if HAS_TENSORFLOW and os.path.exists('ics_autoencoder.h5'):
            try:
                autoencoder = keras.models.load_model('ics_autoencoder.h5', compile=False)
            except Exception as e:
                st.warning(f"⚠️ Could not load autoencoder: {e}")
        
        # Try to load SHAP explainer
        explainer = None
        if HAS_SHAP and os.path.exists('shap_explainer.pkl'):
            try:
                explainer = joblib.load('shap_explainer.pkl')
            except Exception as e:
                st.warning(f"⚠️ Could not load SHAP explainer: {e}")
        
        return xgb_model, scaler, autoencoder, explainer, feature_names
        
    except Exception as e:
        st.error(f"### ⚠️ Error Loading Models: {str(e)}")
        return None, None, None, None, None

# Load models
xgb_model, scaler, autoencoder, shap_explainer, feature_names = load_models()

# Enhanced feature engineering
def engineer_features(df):
    """Apply comprehensive feature engineering"""
    # Basic derived features
    df['energy_residual'] = df['power_input'] - df['power_output']
    df['efficiency'] = df['power_output'] / (df['power_input'] + 1)
    df['temp_flow_ratio'] = df['temperature'] / (df['flow_rate'] + 1)
    df['pressure_flow_ratio'] = df['pressure'] / (df['flow_rate']**2 + 1)
    df['control_mismatch'] = np.abs(df['valve_position'] - df['pump_speed'])
    
    # Rolling statistics (for single row, use previous values or defaults)
    window_size = 10
    if len(df) >= window_size:
        df['flow_rate_rolling_mean'] = df['flow_rate'].rolling(window=window_size, min_periods=1).mean()
        df['flow_rate_rolling_std'] = df['flow_rate'].rolling(window=window_size, min_periods=1).std()
        df['temperature_rolling_mean'] = df['temperature'].rolling(window=window_size, min_periods=1).mean()
        df['pressure_rolling_std'] = df['pressure'].rolling(window=window_size, min_periods=1).std()
    else:
        # For single predictions, use the current values
        df['flow_rate_rolling_mean'] = df['flow_rate']
        df['flow_rate_rolling_std'] = 0
        df['temperature_rolling_mean'] = df['temperature']
        df['pressure_rolling_std'] = 0
    
    # Rate of change (for single row, set to 0)
    df['flow_rate_change'] = 0
    df['temperature_change'] = 0
    df['pressure_change'] = 0
    
    # Interaction features
    df['valve_pump_interaction'] = df['valve_position'] * df['pump_speed']
    df['pressure_temp_interaction'] = df['pressure'] * df['temperature'] / 1000
    
    # Statistical features
    df['power_ratio'] = df['power_output'] / (df['power_input'] + 1)
    df['flow_pressure_product'] = df['flow_rate'] * df['pressure'] / 1000
    
    # Handle edge cases
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)
    
    return df

# Enhanced prediction with SHAP and rule-based boost
def predict_with_explanation(input_data):
    """Make prediction and generate comprehensive SHAP explanation"""
    try:
        # Create DataFrame
        df = pd.DataFrame([input_data])
        
        # Engineer features
        df = engineer_features(df)
        
        # Make prediction
        prediction = xgb_model.predict(df)[0]
        prediction_proba = xgb_model.predict_proba(df)[0]
        
        # Apply rule-based boost for extreme values
        extreme_score = 0
        
        # Check for extreme values
        if input_data['temperature'] < 35 or input_data['temperature'] > 70:
            extreme_score += 0.3
        if input_data['pressure'] < 380 or input_data['pressure'] > 620:
            extreme_score += 0.3
        if input_data['flow_rate'] < 85 or input_data['flow_rate'] > 135:
            extreme_score += 0.2
        
        efficiency = input_data['power_output'] / (input_data['power_input'] + 1)
        if efficiency < 0.70 or efficiency > 0.95:
            extreme_score += 0.3
        
        # If extreme values detected and model says normal, boost to anomaly
        if extreme_score >= 0.5 and prediction == 0:
            st.info(f"🔍 **Rule-Based Override**: Extreme values detected (score: {extreme_score:.2f})")
            # Adjust probabilities
            prediction_proba = np.array([
                max(0.1, prediction_proba[0] - 0.4),  # Reduce normal
                min(0.8, prediction_proba[1] + 0.3),  # Boost anomaly
                prediction_proba[2] + 0.1             # Slight boost fault
            ])
            # Renormalize
            prediction_proba = prediction_proba / prediction_proba.sum()
            prediction = np.argmax(prediction_proba)
        
        # Generate SHAP values
        shap_values = None
        if HAS_SHAP and shap_explainer is not None:
            try:
                shap_values = shap_explainer.shap_values(df)
                
                # Ensure proper format
                if isinstance(shap_values, list):
                    # Multi-class case - ensure each is 2D
                    shap_values = [np.atleast_2d(sv) for sv in shap_values]
                else:
                    # Single output - make 2D
                    shap_values = np.atleast_2d(shap_values)
                    
            except Exception as e:
                st.warning(f"⚠️ SHAP computation warning: {str(e)[:100]}")
                shap_values = None
        
        return prediction, prediction_proba, shap_values, df
        
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None, None, None, None

# Generate detailed explanations
def generate_detailed_explanation(feature, shap_value, feature_value, prediction_class):
    """Generate human-readable, context-aware explanations"""
    
    impact_direction = "INCREASES" if shap_value > 0 else "DECREASES"
    class_names = {0: "Normal", 1: "Anomaly", 2: "Critical Fault"}
    
    explanations = {
        'energy_residual': {
            'description': f"Energy loss between input ({feature_value:.1f} kW difference)",
            'impact': f"High energy residual {impact_direction} {class_names[prediction_class]} likelihood",
            'details': "Energy loss indicates inefficiency or system degradation. Normal range: 50-100 kW.",
            'action': "Check for mechanical wear, thermal losses, or electrical inefficiencies."
        },
        'efficiency': {
            'description': f"System efficiency ({feature_value*100:.1f}%)",
            'impact': f"Current efficiency {impact_direction} {class_names[prediction_class]} probability",
            'details': "Efficiency below 80% suggests performance degradation. Optimal: 85-90%.",
            'action': "Inspect power conversion systems, clean filters, check for blockages."
        },
        'temperature': {
            'description': f"Operating temperature ({feature_value:.1f}°C)",
            'impact': f"Temperature {impact_direction} {class_names[prediction_class]} risk",
            'details': "Normal operating range: 45-55°C. High temps indicate cooling issues.",
            'action': "Verify cooling system, check coolant levels, inspect heat exchangers."
        },
        'pressure': {
            'description': f"System pressure ({feature_value:.1f} kPa)",
            'impact': f"Pressure level {impact_direction} {class_names[prediction_class]} likelihood",
            'details': "Normal range: 450-550 kPa. Deviations suggest leaks or restrictions.",
            'action': "Inspect for leaks, check seals, verify pressure sensors, examine valves."
        },
        'flow_rate': {
            'description': f"Flow rate ({feature_value:.1f} L/min)",
            'impact': f"Flow rate {impact_direction} {class_names[prediction_class]} probability",
            'details': "Optimal: 95-105 L/min. Low flow suggests blockage, high flow indicates bypass.",
            'action': "Clean filters, check for obstructions, verify pump operation."
        },
        'control_mismatch': {
            'description': f"Valve-Pump coordination mismatch ({feature_value:.2f})",
            'impact': f"Control mismatch {impact_direction} {class_names[prediction_class]} risk",
            'details': "Large mismatch indicates poor control coordination or actuator failure.",
            'action': "Calibrate control systems, check actuator response, verify PLC logic."
        },
        'temp_flow_ratio': {
            'description': f"Temperature-Flow relationship ({feature_value:.3f})",
            'impact': f"Heat transfer efficiency {impact_direction} {class_names[prediction_class]} likelihood",
            'details': "Abnormal ratio suggests heat transfer problems or flow restrictions.",
            'action': "Inspect heat exchangers, check flow distribution, verify temperature sensors."
        },
        'pressure_flow_ratio': {
            'description': f"Pressure-Flow dynamics ({feature_value:.4f})",
            'impact': f"Flow dynamics {impact_direction} {class_names[prediction_class]} probability",
            'details': "Deviation from quadratic relationship indicates system anomalies.",
            'action': "Check for partial blockages, verify pump curves, inspect piping."
        }
    }
    
    # Default explanation
    default = {
        'description': f"{feature}: {feature_value:.3f}",
        'impact': f"This feature {impact_direction} {class_names[prediction_class]} prediction",
        'details': f"SHAP value: {shap_value:.4f}",
        'action': "Monitor this parameter closely for trends."
    }
    
    return explanations.get(feature, default)

# Generate recommendations
def generate_comprehensive_recommendations(prediction, shap_analysis, input_data):
    """Generate prioritized, actionable recommendations"""
    recommendations = []
    
    if prediction == 2:  # Critical Fault
        recommendations.append({
            'priority': 'CRITICAL',
            'action': '🚨 IMMEDIATE ACTION REQUIRED',
            'details': 'Initiate emergency shutdown procedure per safety protocol SOP-001',
            'timeline': 'IMMEDIATE'
        })
        recommendations.append({
            'priority': 'CRITICAL',
            'action': 'Isolate Affected Systems',
            'details': 'Prevent cascading failures by isolating the affected control loop',
            'timeline': 'Within 5 minutes'
        })
        recommendations.append({
            'priority': 'HIGH',
            'action': 'Contact Maintenance Team',
            'details': 'Alert on-call maintenance personnel for immediate response',
            'timeline': 'IMMEDIATE'
        })
    
    elif prediction == 1:  # Anomaly
        recommendations.append({
            'priority': 'HIGH',
            'action': '⚠️ Increase Monitoring Frequency',
            'details': 'Monitor system every 2 minutes for the next hour',
            'timeline': 'Next 60 minutes'
        })
        recommendations.append({
            'priority': 'MEDIUM',
            'action': 'Log Event for Analysis',
            'details': 'Record this anomaly in the maintenance log for trend analysis',
            'timeline': 'Within 15 minutes'
        })
    
    # Feature-specific recommendations based on SHAP
    if shap_analysis:
        top_feature = shap_analysis[0]
        feature_name = top_feature['feature']
        
        if 'temperature' in feature_name.lower():
            if input_data.get('temperature', 50) > 60:
                recommendations.append({
                    'priority': 'HIGH',
                    'action': 'Address High Temperature',
                    'details': f"Temperature at {input_data['temperature']:.1f}°C (threshold: 60°C). Check cooling system immediately.",
                    'timeline': 'Within 30 minutes'
                })
        
        if 'pressure' in feature_name.lower():
            if input_data.get('pressure', 500) < 450:
                recommendations.append({
                    'priority': 'HIGH',
                    'action': 'Investigate Low Pressure',
                    'details': f"Pressure at {input_data['pressure']:.1f} kPa (minimum: 450 kPa). Check for leaks.",
                    'timeline': 'Within 20 minutes'
                })
        
        if 'efficiency' in feature_name.lower() or 'energy' in feature_name.lower():
            efficiency = input_data.get('power_output', 400) / (input_data.get('power_input', 500) + 1)
            if efficiency < 0.80:
                recommendations.append({
                    'priority': 'MEDIUM',
                    'action': 'Improve System Efficiency',
                    'details': f"Current efficiency: {efficiency*100:.1f}% (target: >80%). Schedule maintenance inspection.",
                    'timeline': 'Within 24 hours'
                })
        
        if 'control' in feature_name.lower() or 'mismatch' in feature_name.lower():
            recommendations.append({
                'priority': 'MEDIUM',
                'action': 'Calibrate Control Systems',
                'details': 'Valve-pump coordination is suboptimal. Recalibrate control parameters.',
                'timeline': 'Within 48 hours'
            })
    
    # Default recommendations
    if prediction == 0:
        recommendations.append({
            'priority': 'LOW',
            'action': '✅ Continue Normal Operations',
            'details': 'All parameters within normal ranges. Maintain regular monitoring schedule.',
            'timeline': 'Ongoing'
        })
    
    return recommendations

# Main app
def main():
    # Header
    st.markdown('<p class="main-header">🛡️ Enhanced ICS/SCADA Anomaly Detection System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Industrial Control System Monitoring with Advanced SHAP Explainability</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100)
        st.title("⚙️ System Configuration")
        
        st.markdown("---")
        st.info("""
        📊 **Model Information**
        
        - **Architecture**: Enhanced Autoencoder + Optimized XGBoost
        - **Features**: 20+ physics-aware features
        - **Training Samples**: 30,000+
        - **Classes**: Normal, Anomaly, Critical Fault
        """)
        
        st.markdown("---")
        mode = st.radio(
            "🎯 Operating Mode",
            ["Single Prediction", "Batch Analysis", "Live Monitoring"],
            help="Choose your analysis mode"
        )
        
        st.markdown("---")
        if st.button("📖 View Documentation", use_container_width=True):
            st.session_state.show_docs = True
    
    # Mode routing
    if mode == "Single Prediction":
        single_prediction_mode()
    elif mode == "Batch Analysis":
        batch_analysis_mode()
    else:
        live_monitoring_mode()

def single_prediction_mode():
    """Enhanced single prediction interface with comprehensive SHAP"""
    st.header("🔍 Single System Analysis")
    
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.subheader("📥 Input Parameters")
        
        with st.expander("⚙️ Process Variables", expanded=True):
            flow_rate = st.slider("Flow Rate (L/min)", 80.0, 140.0, 100.0, 0.5,
                                help="Optimal range: 95-105 L/min")
            pressure = st.slider("Pressure (kPa)", 300.0, 700.0, 500.0, 5.0,
                               help="Normal range: 450-550 kPa")
            temperature = st.slider("Temperature (°C)", 30.0, 80.0, 50.0, 0.5,
                                  help="Optimal range: 45-55°C")
        
        with st.expander("⚡ Power System", expanded=True):
            power_input = st.slider("Power Input (kW)", 400.0, 600.0, 500.0, 5.0,
                                  help="Nominal: 500 kW")
            power_output = st.slider("Power Output (kW)", 300.0, 550.0, 425.0, 5.0,
                                   help="Expected: 85% of input")
        
        with st.expander("🎛️ Control Parameters", expanded=True):
            valve_position = st.slider("Valve Position (%)", 0.0, 100.0, 60.0, 0.5,
                                     help="Current valve opening") / 100
            pump_speed = st.slider("Pump Speed (%)", 0.0, 100.0, 70.0, 0.5,
                                 help="Current pump speed") / 100
        
        # Quick presets
        st.markdown("---")
        st.subheader("🎯 Quick Presets")
        col_p1, col_p2, col_p3 = st.columns(3)
        
        if col_p1.button("✅ Normal", use_container_width=True):
            st.session_state.preset = 'normal'
        if col_p2.button("⚠️ Anomaly", use_container_width=True):
            st.session_state.preset = 'anomaly'
        if col_p3.button("🚨 Fault", use_container_width=True):
            st.session_state.preset = 'fault'
        
        st.markdown("---")
        analyze_btn = st.button("🚀 Run Anomaly Detection", 
                               use_container_width=True,
                               type="primary")
    
    with col2:
        st.subheader("📊 Analysis Results")
        
        if analyze_btn or 'preset' in st.session_state:
            # Handle presets with more extreme values
            if 'preset' in st.session_state:
                if st.session_state.preset == 'anomaly':
                    temperature = 72.0  # Higher than before
                    power_output = 365.0  # Lower efficiency
                    pressure = 420.0  # Lower pressure
                    st.info("🎯 Loaded **Anomaly** preset with extreme values")
                elif st.session_state.preset == 'fault':
                    temperature = 78.0  # Very high
                    pressure = 340.0  # Very low
                    power_output = 310.0  # Very low
                    flow_rate = 85.0  # Low flow
                    st.error("🎯 Loaded **Critical Fault** preset with extreme values")
                del st.session_state.preset
            
            with st.spinner("🔄 Analyzing system parameters..."):
                # Prepare input
                input_data = {
                    'flow_rate': flow_rate,
                    'pressure': pressure,
                    'temperature': temperature,
                    'power_input': power_input,
                    'power_output': power_output,
                    'valve_position': valve_position,
                    'pump_speed': pump_speed
                }
                
                # Make prediction
                prediction, proba, shap_values, df = predict_with_explanation(input_data)
                
                if prediction is not None:
                    # Display status
                    status_names = ['✅ NORMAL OPERATION', '⚠️ ANOMALY DETECTED', '🚨 CRITICAL FAULT']
                    status_colors = ['alert-normal', 'alert-warning', 'alert-critical']
                    
                    st.markdown(f'<div class="{status_colors[prediction]}">{status_names[prediction]}</div>', 
                               unsafe_allow_html=True)
                    
                    # Confidence metrics
                    st.markdown("---")
                    st.markdown("### 📈 Confidence Scores")
                    col_conf1, col_conf2, col_conf3 = st.columns(3)
                    
                    with col_conf1:
                        st.metric("Normal", f"{proba[0]*100:.1f}%",
                                delta=f"{(proba[0]-0.33)*100:+.1f}%" if proba[0] != max(proba) else None)
                    with col_conf2:
                        st.metric("Anomaly", f"{proba[1]*100:.1f}%",
                                delta=f"{(proba[1]-0.33)*100:+.1f}%" if proba[1] != max(proba) else None)
                    with col_conf3:
                        st.metric("Fault", f"{proba[2]*100:.1f}%",
                                delta=f"{(proba[2]-0.33)*100:+.1f}%" if proba[2] != max(proba) else None)
                    
                    # Confidence gauge
                    max_proba = max(proba)
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=max_proba * 100,
                        title={'text': f"Confidence in {status_names[prediction].split()[1]}"},
                        delta={'reference': 80},
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': ["#00C851", "#ffbb33", "#ff4444"][prediction]},
                            'steps': [
                                {'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 80], 'color': "lightyellow"},
                                {'range': [80, 100], 'color': "lightgreen"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 90
                            }
                        }
                    ))
                    fig_gauge.update_layout(height=300)
                    st.plotly_chart(fig_gauge, use_container_width=True)
                    
                    # SHAP Explanation
                    st.markdown("---")
                    st.markdown("### 🔬 Root Cause Analysis (SHAP)")
                    
                    # Initialize top_shap as None for scope
                    top_shap = None
                    shap_analysis = None
                    
                    if HAS_SHAP and shap_values is not None:
                        try:
                            # Extract SHAP values for predicted class
                            if isinstance(shap_values, list):
                                if len(shap_values) > prediction:
                                    shap_vals = shap_values[prediction]
                                else:
                                    shap_vals = shap_values[0]
                            else:
                                shap_vals = shap_values
                            
                            # Convert to numpy array and ensure proper shape
                            shap_vals = np.array(shap_vals)
                            
                            # Flatten to 1D if needed
                            if shap_vals.ndim > 1:
                                shap_vals_flat = shap_vals.flatten()
                            else:
                                shap_vals_flat = shap_vals
                            
                            # Prepare SHAP data
                            feature_names_list = df.columns.tolist()
                            
                            # Ensure same length
                            if len(shap_vals_flat) != len(feature_names_list):
                                # Take first row if 2D
                                if shap_vals.ndim > 1:
                                    shap_vals_flat = shap_vals[0]
                                # Or pad/truncate to match
                                min_len = min(len(shap_vals_flat), len(feature_names_list))
                                shap_vals_flat = shap_vals_flat[:min_len]
                                feature_names_list = feature_names_list[:min_len]
                            
                            shap_importance = np.abs(shap_vals_flat)
                            
                            shap_df = pd.DataFrame({
                                'Feature': feature_names_list,
                                'SHAP Value': shap_vals_flat,
                                'Absolute Impact': shap_importance,
                                'Feature Value': df.iloc[0, :len(shap_vals_flat)].values
                            }).sort_values('Absolute Impact', ascending=False)
                            
                            # Top 10 features
                            top_shap = shap_df.head(10)
                            shap_analysis = top_shap.head(3).to_dict('records')
                            
                            # Interactive SHAP plot
                            fig_shap = go.Figure()
                            
                            # Convert SHAP values to list of floats
                            shap_values_list = [float(x) for x in top_shap['SHAP Value'].values]
                            abs_impact_list = [float(x) for x in top_shap['Absolute Impact'].values]
                            
                            fig_shap.add_trace(go.Bar(
                                y=top_shap['Feature'].tolist(),
                                x=abs_impact_list,
                                orientation='h',
                                marker=dict(
                                    color=shap_values_list,
                                    colorscale='RdBu_r',
                                    showscale=True,
                                    colorbar=dict(title="SHAP<br>Impact"),
                                    cmin=-max(abs(x) for x in shap_values_list),
                                    cmax=max(abs(x) for x in shap_values_list)
                                ),
                                text=[f"{val:.3f}" for val in shap_values_list],
                                textposition='outside',
                                hovertemplate='<b>%{y}</b><br>Impact: %{x:.4f}<br>SHAP: %{text}<extra></extra>'
                            ))
                            
                            fig_shap.update_layout(
                                title='Top 10 Contributing Features (SHAP Analysis)',
                                xaxis_title='Absolute SHAP Impact',
                                yaxis_title='Feature',
                                height=500,
                                yaxis=dict(autorange="reversed")
                            )
                            
                            st.plotly_chart(fig_shap, use_container_width=True)
                            
                            # Detailed explanations
                            st.markdown("### 📋 Detailed Feature Analysis")
                            
                            for idx, row in top_shap.head(5).iterrows():
                                feature = row['Feature']
                                impact = float(row['Absolute Impact'])
                                shap_val = float(row['SHAP Value'])
                                feat_val = float(row['Feature Value'])
                                
                                explanation = generate_detailed_explanation(
                                    feature, shap_val, feat_val, prediction
                                )
                                
                                is_first = (idx == top_shap.index[0])
                                with st.expander(f"**{feature}** - Impact Score: {impact:.4f}", expanded=is_first):
                                    st.markdown(f"**{explanation['description']}**")
                                    st.info(f"💡 **Impact**: {explanation['impact']}")
                                    st.markdown(f"📖 **Details**: {explanation['details']}")
                                    st.warning(f"🔧 **Recommended Action**: {explanation['action']}")
                                    
                                    # Visual impact indicator
                                    impact_normalized = min(100, impact * 1000)
                                    st.progress(impact_normalized / 100)
                            
                            # SHAP waterfall - simplified
                            st.markdown("### 🌊 SHAP Impact Breakdown")
                            
                            # Create simple waterfall with Plotly
                            fig_waterfall = go.Figure(go.Waterfall(
                                name="SHAP",
                                orientation="v",
                                measure=["relative"] * len(top_shap.head(8)),
                                x=top_shap.head(8)['Feature'].tolist(),
                                y=shap_values_list[:8],
                                connector={"line": {"color": "rgb(63, 63, 63)"}},
                                decreasing={"marker": {"color": "#3D9970"}},
                                increasing={"marker": {"color": "#FF4136"}},
                            ))
                            
                            fig_waterfall.update_layout(
                                title="How Features Push the Prediction",
                                xaxis_title="Feature",
                                yaxis_title="SHAP Value",
                                height=400
                            )
                            
                            st.plotly_chart(fig_waterfall, use_container_width=True)
                        
                        except Exception as e:
                            st.warning(f"⚠️ SHAP visualization error: {str(e)[:150]}")
                            st.info("📊 Showing XGBoost feature importance instead")
                            # Fallback to simple feature importance
                            st.info("📊 Showing XGBoost feature importance instead")
                            feature_imp = pd.DataFrame({
                                'feature': df.columns,
                                'importance': xgb_model.feature_importances_
                            }).sort_values('importance', ascending=False).head(10)
                            
                            fig_imp = px.bar(feature_imp, x='importance', y='feature', 
                                           orientation='h', title='Top 10 Feature Importances')
                            st.plotly_chart(fig_imp, use_container_width=True)
                            
                    else:
                        # SHAP not available - use feature importance
                        st.warning("⚠️ SHAP explainability not available")
                        st.info("📊 Showing XGBoost feature importance instead")
                        
                        feature_imp = pd.DataFrame({
                            'feature': df.columns,
                            'importance': xgb_model.feature_importances_
                        }).sort_values('importance', ascending=False).head(10)
                        
                        fig_imp = px.bar(feature_imp, x='importance', y='feature',
                                       orientation='h', 
                                       title='Top 10 Feature Importances',
                                       color='importance',
                                       color_continuous_scale='Blues')
                        st.plotly_chart(fig_imp, use_container_width=True)
                        
                        # Rule-based analysis
                        st.markdown("### 📋 Rule-Based Analysis")
                        issues = []
                        
                        if input_data['temperature'] > 65:
                            issues.append(("🌡️ High Temperature", input_data['temperature'], 
                                         "Temperature exceeds safe range (>65°C)"))
                        if input_data['temperature'] < 40:
                            issues.append(("🌡️ Low Temperature", input_data['temperature'],
                                         "Temperature below optimal range (<40°C)"))
                        if input_data['pressure'] < 420:
                            issues.append(("📉 Low Pressure", input_data['pressure'], 
                                         "Pressure below minimum threshold (<420 kPa)"))
                        if input_data['pressure'] > 600:
                            issues.append(("📈 High Pressure", input_data['pressure'],
                                         "Pressure exceeds safe limit (>600 kPa)"))
                        
                        efficiency = input_data['power_output'] / (input_data['power_input'] + 1)
                        if efficiency < 0.75:
                            issues.append(("⚡ Low Efficiency", efficiency, 
                                         "Power efficiency degraded (<75%)"))
                        
                        flow_deviation = abs(input_data['flow_rate'] - 100) / 100
                        if flow_deviation > 0.15:
                            issues.append(("💧 Flow Deviation", input_data['flow_rate'],
                                         f"Flow rate {flow_deviation*100:.0f}% from nominal"))
                        
                        control_diff = abs(input_data['valve_position'] - input_data['pump_speed'])
                        if control_diff > 0.3:
                            issues.append(("🎛️ Control Mismatch", control_diff,
                                         "Large valve-pump coordination gap"))
                        
                        if issues:
                            for issue, value, desc in issues:
                                st.warning(f"**{issue}**: {value:.2f} - {desc}")
                        else:
                            st.success("✅ No obvious issues detected in parameters")
                    
                    # Recommendations - now safe to use top_shap
                    if prediction > 0:
                        st.markdown("---")
                        st.markdown("### 💡 Recommended Actions")
                        
                        recommendations = generate_comprehensive_recommendations(
                            prediction, shap_analysis, input_data
                        )
                        # Extract SHAP values for predicted class
                        if isinstance(shap_values, list):
                            shap_vals = shap_values[prediction]
                        else:
                            shap_vals = shap_values
                        
                        # Ensure shap_vals is 2D array
                        if len(shap_vals.shape) == 1:
                            shap_vals = shap_vals.reshape(1, -1)
                        
                        # Prepare SHAP data
                        feature_names_list = df.columns.tolist()
                        
                        # Flatten SHAP values if needed
                        shap_vals_flat = shap_vals[0] if shap_vals.shape[0] > 0 else shap_vals.flatten()
                        shap_importance = np.abs(shap_vals_flat)
                        
                        shap_df = pd.DataFrame({
                            'Feature': feature_names_list,
                            'SHAP Value': shap_vals_flat,
                            'Absolute Impact': shap_importance,
                            'Feature Value': df.iloc[0].values
                        }).sort_values('Absolute Impact', ascending=False)
                        
                        # Top 10 features
                        top_shap = shap_df.head(10)
                        
                        # Interactive SHAP plot
                        fig_shap = go.Figure()
                        
                        # Convert SHAP values to float for comparison
                        shap_values_for_color = top_shap['SHAP Value'].values
                        colors = ['red' if float(x) > 0 else 'blue' for x in shap_values_for_color]
                        
                        fig_shap.add_trace(go.Bar(
                            y=top_shap['Feature'],
                            x=top_shap['Absolute Impact'],
                            orientation='h',
                            marker=dict(
                                color=top_shap['SHAP Value'],
                                colorscale='RdBu_r',
                                showscale=True,
                                colorbar=dict(title="SHAP<br>Impact")
                            ),
                            text=[f"{val:.3f}" for val in top_shap['SHAP Value']],
                            textposition='outside',
                            hovertemplate='<b>%{y}</b><br>Impact: %{x:.4f}<br>SHAP: %{text}<extra></extra>'
                        ))
                        
                        fig_shap.update_layout(
                            title='Top 10 Contributing Features (SHAP Analysis)',
                            xaxis_title='Absolute SHAP Impact',
                            yaxis_title='Feature',
                            height=500,
                            yaxis=dict(autorange="reversed")
                        )
                        
                        st.plotly_chart(fig_shap, use_container_width=True)
                        
                        # Detailed explanations
                        st.markdown("### 📋 Detailed Feature Analysis")
                        
                        for idx, row in top_shap.head(5).iterrows():
                            feature = row['Feature']
                            impact = row['Absolute Impact']
                            shap_val = row['SHAP Value']
                            feat_val = row['Feature Value']
                            
                            explanation = generate_detailed_explanation(
                                feature, shap_val, feat_val, prediction
                            )
                            
                            with st.expander(f"**{feature}** - Impact Score: {impact:.4f}", expanded=idx==top_shap.index[0]):
                                st.markdown(f"**{explanation['description']}**")
                                st.info(f"💡 **Impact**: {explanation['impact']}")
                                st.markdown(f"📖 **Details**: {explanation['details']}")
                                st.warning(f"🔧 **Recommended Action**: {explanation['action']}")
                                
                                # Visual impact indicator
                                impact_normalized = min(100, impact * 1000)
                                st.progress(impact_normalized / 100)
                        
                        # SHAP waterfall for single prediction
                        try:
                            st.markdown("### 🌊 SHAP Waterfall Plot")
                            st.markdown("*Shows how each feature pushes the prediction from base value*")
                            
                            fig, ax = plt.subplots(figsize=(10, 8))
                            
                            # Prepare explanation object based on SHAP values structure
                            if isinstance(shap_values, list):
                                expected_val = shap_explainer.expected_value[prediction] if isinstance(shap_explainer.expected_value, (list, np.ndarray)) else shap_explainer.expected_value
                                shap_vals_for_plot = shap_values[prediction][0] if len(shap_values[prediction].shape) > 1 else shap_values[prediction]
                            else:
                                expected_val = shap_explainer.expected_value if isinstance(shap_explainer.expected_value, (int, float)) else shap_explainer.expected_value[0]
                                shap_vals_for_plot = shap_vals_flat
                            
                            explanation = shap.Explanation(
                                values=shap_vals_for_plot,
                                base_values=expected_val,
                                data=df.iloc[0].values,
                                feature_names=feature_names_list
                            )
                            
                            shap.plots.waterfall(explanation, max_display=12, show=False)
                            st.pyplot(fig)
                            plt.close()
                        except Exception as e:
                            st.info(f"ℹ️ Waterfall plot unavailable. Using bar chart instead.")
                            # Show alternative visualization
                            fig_alt = px.bar(
                                top_shap.head(8),
                                y='Feature',
                                x='SHAP Value',
                                orientation='h',
                                title='Feature Impact on Prediction',
                                color='SHAP Value',
                                color_continuous_scale='RdBu_r'
                            )
                            st.plotly_chart(fig_alt, use_container_width=True)
                        
                    else:
                        st.warning("⚠️ SHAP explainability not available")
                        st.info("Install SHAP library for detailed explanations: `pip install shap`")
                    
                    # Recommendations
                    if prediction > 0:
                        st.markdown("---")
                        st.markdown("### 💡 Recommended Actions")
                        
                        # Generate recommendations
                        shap_analysis = None
                        if HAS_SHAP and shap_values is not None:
                            shap_analysis = top_shap.head(3).to_dict('records')
                        
                        recommendations = generate_comprehensive_recommendations(
                            prediction, shap_analysis, input_data
                        )
                        
                        priority_colors = {
                            'CRITICAL': '🔴',
                            'HIGH': '🟠',
                            'MEDIUM': '🟡',
                            'LOW': '🟢'
                        }
                        
                        for i, rec in enumerate(recommendations):
                            priority = rec.get('priority', 'LOW')
                            color_emoji = priority_colors.get(priority, '⚪')
                            
                            st.markdown(f"#### {color_emoji} Priority: {priority}")
                            st.markdown(f"**Action**: {rec['action']}")
                            st.markdown(f"**Details**: {rec['details']}")
                            st.markdown(f"**Timeline**: {rec['timeline']}")
                            if i < len(recommendations) - 1:
                                st.markdown("---")
                
                else:
                    st.error("Failed to generate prediction. Please check model files.")

def batch_analysis_mode():
    """Enhanced batch file analysis"""
    st.header("📂 Batch Analysis Mode")
    
    st.info("📤 Upload a CSV file containing ICS data with the required columns")
    
    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=['csv'],
        help="File must contain: flow_rate, pressure, temperature, power_input, power_output, valve_position, pump_speed"
    )
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ Loaded {len(df)} samples")
            
            # Data preview
            with st.expander("📊 Data Preview", expanded=True):
                st.dataframe(df.head(20), use_container_width=True)
                
                # Basic statistics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Samples", len(df))
                col2.metric("Features", len(df.columns))
                col3.metric("Missing Values", df.isnull().sum().sum())
                col4.metric("Duplicates", df.duplicated().sum())
            
            # Analyze button
            if st.button("🚀 Analyze Batch", type="primary", use_container_width=True):
                with st.spinner("Processing batch..."):
                    required_cols = ['flow_rate', 'pressure', 'temperature', 'power_input',
                                   'power_output', 'valve_position', 'pump_speed']
                    
                    if all(col in df.columns for col in required_cols):
                        # Process data
                        df_processed = df[required_cols].copy()
                        df_processed = engineer_features(df_processed)
                        
                        # Make predictions
                        predictions = xgb_model.predict(df_processed)
                        probabilities = xgb_model.predict_proba(df_processed)
                        
                        # Add results to dataframe
                        df['Prediction'] = predictions
                        df['Status'] = df['Prediction'].map({0: 'Normal', 1: 'Anomaly', 2: 'Fault'})
                        df['Confidence'] = [probs[pred] for pred, probs in zip(predictions, probabilities)]
                        
                        # Statistics
                        st.markdown("---")
                        st.markdown("### 📊 Detection Results")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        normal_count = (predictions == 0).sum()
                        anomaly_count = (predictions == 1).sum()
                        fault_count = (predictions == 2).sum()
                        
                        col1.metric("✅ Normal", normal_count, 
                                  f"{normal_count/len(df)*100:.1f}%")
                        col2.metric("⚠️ Anomalies", anomaly_count,
                                  f"{anomaly_count/len(df)*100:.1f}%")
                        col3.metric("🚨 Faults", fault_count,
                                  f"{fault_count/len(df)*100:.1f}%")
                        col4.metric("📈 Avg Confidence", 
                                  f"{df['Confidence'].mean()*100:.1f}%")
                        
                        # Visualizations
                        col_viz1, col_viz2 = st.columns(2)
                        
                        with col_viz1:
                            # Pie chart
                            fig_pie = px.pie(
                                df, 
                                names='Status',
                                title='Detection Distribution',
                                color='Status',
                                color_discrete_map={
                                    'Normal': '#00C851',
                                    'Anomaly': '#ffbb33',
                                    'Fault': '#ff4444'
                                }
                            )
                            st.plotly_chart(fig_pie, use_container_width=True)
                        
                        with col_viz2:
                            # Time series if index is available
                            fig_line = go.Figure()
                            
                            for status, color in [('Normal', '#00C851'), 
                                                ('Anomaly', '#ffbb33'), 
                                                ('Fault', '#ff4444')]:
                                mask = df['Status'] == status
                                fig_line.add_trace(go.Scatter(
                                    x=df[mask].index,
                                    y=df[mask]['Confidence'] * 100,
                                    mode='markers',
                                    name=status,
                                    marker=dict(color=color, size=8)
                                ))
                            
                            fig_line.update_layout(
                                title='Confidence Over Time',
                                xaxis_title='Sample Index',
                                yaxis_title='Confidence (%)',
                                height=400
                            )
                            st.plotly_chart(fig_line, use_container_width=True)
                        
                        # Detailed results
                        with st.expander("📋 Detailed Results", expanded=False):
                            st.dataframe(
                                df[['flow_rate', 'pressure', 'temperature', 
                                   'Status', 'Confidence']].head(50),
                                use_container_width=True
                            )
                        
                        # Anomalies and faults details
                        if anomaly_count > 0 or fault_count > 0:
                            st.markdown("---")
                            st.markdown("### ⚠️ Flagged Samples")
                            
                            flagged = df[df['Prediction'] > 0].copy()
                            flagged = flagged.sort_values('Confidence', ascending=False)
                            
                            st.dataframe(
                                flagged[['flow_rate', 'pressure', 'temperature',
                                       'power_input', 'power_output', 'Status', 'Confidence']],
                                use_container_width=True
                            )
                        
                        # Download results
                        st.markdown("---")
                        csv = df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Complete Results",
                            csv,
                            "anomaly_detection_results.csv",
                            "text/csv",
                            use_container_width=True
                        )
                    else:
                        missing = [col for col in required_cols if col not in df.columns]
                        st.error(f"❌ Missing required columns: {', '.join(missing)}")
                        
        except Exception as e:
            st.error(f"Error processing file: {e}")

def live_monitoring_mode():
    """Simulated live monitoring dashboard"""
    st.header("📡 Live Monitoring Dashboard")
    
    st.info("🔴 **LIVE** - Simulated real-time monitoring (auto-refresh every 3 seconds)")
    
    # Controls
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("▶️ Start Monitoring", use_container_width=True, type="primary"):
            st.session_state.monitoring = True
            st.session_state.monitoring_data = []
    
    with col2:
        if st.button("⏹️ Stop Monitoring", use_container_width=True):
            st.session_state.monitoring = False
    
    with col3:
        refresh_rate = st.slider("Refresh Rate (seconds)", 1, 10, 3)
    
    # Monitoring display
    if st.session_state.get('monitoring', False):
        placeholder = st.empty()
        
        # Initialize monitoring data
        if 'monitoring_data' not in st.session_state:
            st.session_state.monitoring_data = []
        
        for iteration in range(30):  # Monitor for 30 iterations
            if not st.session_state.get('monitoring', False):
                break
            
            with placeholder.container():
                # Generate simulated data
                base_temp = 50 + 10 * np.sin(iteration / 5)
                base_flow = 100 + 5 * np.cos(iteration / 3)
                
                # Add anomalies occasionally
                is_anomaly = np.random.random() < 0.15
                is_fault = np.random.random() < 0.05
                
                data = {
                    'flow_rate': base_flow + np.random.normal(0, 2),
                    'pressure': 500 + np.random.normal(0, 20),
                    'temperature': base_temp + (15 if is_anomaly else 0) + np.random.normal(0, 2),
                    'power_input': 500 + np.random.normal(0, 10),
                    'power_output': 425 * (0.7 if is_fault else 1.0) + np.random.normal(0, 10),
                    'valve_position': 0.6 + np.random.normal(0, 0.05),
                    'pump_speed': 0.7 + np.random.normal(0, 0.05)
                }
                
                # Predict
                prediction, proba, _, _ = predict_with_explanation(data)
                
                # Store data
                st.session_state.monitoring_data.append({
                    'time': datetime.now(),
                    'prediction': prediction,
                    'confidence': proba[prediction] if prediction is not None else 0,
                    **data
                })
                
                # Keep only last 50 samples
                if len(st.session_state.monitoring_data) > 50:
                    st.session_state.monitoring_data.pop(0)
                
                # Display current status
                col_status1, col_status2, col_status3 = st.columns([1, 1, 2])
                
                with col_status1:
                    st.metric("Timestamp", datetime.now().strftime("%H:%M:%S"))
                    status_map = ['✅ Normal', '⚠️ Anomaly', '🚨 Fault']
                    st.metric("Status", status_map[prediction] if prediction is not None else "Error")
                
                with col_status2:
                    if prediction is not None:
                        st.metric("Confidence", f"{proba[prediction]*100:.1f}%")
                        st.metric("Temperature", f"{data['temperature']:.1f}°C")
                
                with col_status3:
                    # Real-time gauge
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=data['temperature'],
                        title={'text': "Temperature (°C)"},
                        gauge={
                            'axis': {'range': [30, 80]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [30, 50], 'color': "lightgreen"},
                                {'range': [50, 65], 'color': "yellow"},
                                {'range': [65, 80], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 70
                            }
                        }
                    ))
                    fig_gauge.update_layout(height=250)
                    st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Historical chart
                if len(st.session_state.monitoring_data) > 1:
                    st.markdown("---")
                    st.markdown("### 📈 Real-Time Trends")
                    
                    history_df = pd.DataFrame(st.session_state.monitoring_data)
                    
                    fig_trends = go.Figure()
                    
                    fig_trends.add_trace(go.Scatter(
                        x=list(range(len(history_df))),
                        y=history_df['temperature'],
                        mode='lines+markers',
                        name='Temperature',
                        line=dict(color='red', width=2)
                    ))
                    
                    fig_trends.add_trace(go.Scatter(
                        x=list(range(len(history_df))),
                        y=history_df['pressure'] / 10,
                        mode='lines+markers',
                        name='Pressure/10',
                        line=dict(color='blue', width=2)
                    ))
                    
                    fig_trends.update_layout(
                        title='System Parameters Over Time',
                        xaxis_title='Sample',
                        yaxis_title='Value',
                        height=300,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig_trends, use_container_width=True)
                    
                    # Alert summary
                    recent_predictions = [d['prediction'] for d in st.session_state.monitoring_data[-10:]]
                    anomaly_rate = sum(1 for p in recent_predictions if p > 0) / len(recent_predictions)
                    
                    if anomaly_rate > 0.3:
                        st.error(f"⚠️ **HIGH ALERT**: {anomaly_rate*100:.0f}% of recent samples flagged!")
            
            import time
            time.sleep(refresh_rate)
    
    else:
        st.info("👆 Click 'Start Monitoring' to begin live system monitoring")

if __name__ == "__main__":
    if xgb_model and scaler:
        main()
    else:
        st.error("### ⚠️ Models Not Found")
        
        missing = check_model_files()
        if missing:
            st.markdown("**Missing Files:**")
            for item in missing:
                st.markdown(f"- {item}")
        
        st.markdown("---")
        st.markdown("### 🔧 Setup Instructions")
        
        st.code("python enhanced_training_with_shap.py", language="bash")
        
        st.markdown("""
        This will:
        1. Generate enhanced synthetic ICS data (30,000 samples)
        2. Engineer 20+ physics-aware features
        3. Train optimized XGBoost model
        4. Generate SHAP explanations
        5. Create all required files
        
        **Required packages:**
        ```bash
        pip install streamlit pandas numpy scikit-learn xgboost plotly seaborn matplotlib joblib shap tensorflow
        ```
        """)
        
        st.info("💡 After running the training script, refresh this page to start using the dashboard.")