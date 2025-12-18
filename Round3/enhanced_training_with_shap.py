import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# Try to import tensorflow
try:
    from tensorflow import keras
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    HAS_TENSORFLOW = True
except ImportError:
    print("⚠️  TensorFlow not found. Skipping Autoencoder training.")
    HAS_TENSORFLOW = False

# Try to import SHAP
try:
    import shap
    HAS_SHAP = True
except ImportError:
    print("⚠️  SHAP not found. Skipping explainability analysis.")
    HAS_SHAP = False

print("=" * 80)
print("🛡️  ENHANCED ICS/SCADA ANOMALY DETECTION - COMPLETE PIPELINE")
print("=" * 80)

# ============================================================================
# STEP 1: GENERATE ENHANCED SYNTHETIC DATA
# ============================================================================
print("\n[STEP 1/8] Generating Enhanced Synthetic ICS Data...")

np.random.seed(42)
N = 30000  # Increased dataset size

# Initialize arrays
flow = np.zeros(N)
pressure = np.zeros(N)
temperature = np.zeros(N)
power_in = np.zeros(N)
power_out = np.zeros(N)
valve = np.zeros(N)
pump = np.zeros(N)
labels = np.zeros(N, dtype=int)

# Initial conditions
flow[0] = 100
temperature[0] = 50
pressure[0] = 500
power_in[0] = 500
valve[0] = 0.6
pump[0] = 0.7

# Enhanced regime probabilities with better separation
p_anomaly = 0.15
p_fault = 0.08

print(f"  → Generating {N} samples with enhanced temporal dynamics...")

# Generate more distinct patterns
for t in range(1, N):
    # Add cyclic patterns (daily/hourly cycles)
    cycle_factor = np.sin(2 * np.pi * t / 1000) * 0.1
    
    # Control actions with smoother transitions
    valve[t] = np.clip(valve[t-1] + np.random.normal(0, 0.015) + cycle_factor, 0, 1)
    pump[t] = np.clip(pump[t-1] + np.random.normal(0, 0.012) + cycle_factor * 0.5, 0, 1)

    # Enhanced flow dynamics
    flow[t] = (
        95
        + 20 * valve[t]
        + 15 * pump[t]
        + 5 * cycle_factor
        + np.random.normal(0, 2.5)
    )

    # Pressure dynamics (more realistic nonlinear relationship)
    pressure[t] = (
        400
        + 0.035 * flow[t]**2
        + 50 * pump[t]
        + np.random.normal(0, 15)
    )

    # Power input with realistic drift
    power_in[t] = (
        500
        + 0.95 * (power_in[t-1] - 500)
        + np.random.normal(0, 4)
    )

    # Temperature with thermal inertia
    temperature[t] = (
        0.97 * temperature[t-1]
        + 0.03 * (45 + power_in[t] / (flow[t] + 10))
        + np.random.normal(0, 0.6)
    )

    # Efficiency with degradation over time
    efficiency = np.clip(
        0.86 - (t / N) * 0.02 + np.random.normal(0, 0.025),
        0.72, 0.92
    )

    power_out[t] = (
        efficiency * power_in[t]
        + np.random.normal(0, 12)
    )

# Strategic anomaly and fault injection
anomaly_indices = []
fault_indices = []

# Create anomaly clusters (more realistic)
num_anomaly_clusters = int(N * p_anomaly / 20)
for _ in range(num_anomaly_clusters):
    start_idx = np.random.randint(100, N - 100)
    cluster_size = np.random.randint(15, 30)
    anomaly_indices.extend(range(start_idx, min(start_idx + cluster_size, N)))

anomaly_indices = list(set(anomaly_indices))[:int(N * p_anomaly)]

# Create fault events
num_fault_clusters = int(N * p_fault / 15)
for _ in range(num_fault_clusters):
    start_idx = np.random.randint(100, N - 100)
    cluster_size = np.random.randint(10, 25)
    fault_indices.extend(range(start_idx, min(start_idx + cluster_size, N)))

fault_indices = list(set(fault_indices))[:int(N * p_fault)]

print(f"  → Injecting {len(anomaly_indices)} anomalies in {num_anomaly_clusters} clusters...")

# Inject subtle anomalies (deviation from normal but harder to detect)
for idx in anomaly_indices:
    labels[idx] = 1
    deviation = np.random.uniform(0.88, 1.08)
    power_out[idx] *= deviation
    temperature[idx] += np.random.uniform(-3, 5)
    flow[idx] *= np.random.uniform(0.92, 1.05)

print(f"  → Injecting {len(fault_indices)} critical faults in {num_fault_clusters} clusters...")

# Inject clear faults with cascading effects
for idx in fault_indices:
    labels[idx] = 2
    for k in range(idx, min(idx + 8, N)):
        pressure[k] *= np.random.uniform(0.65, 0.82)
        temperature[k] *= np.random.uniform(1.15, 1.45)
        power_out[k] *= np.random.uniform(0.55, 0.75)
        flow[k] *= np.random.uniform(0.85, 0.95)

# Minimal label noise (1%)
noise_idx = np.random.choice(N, int(0.01 * N), replace=False)
labels[noise_idx] = np.random.choice([0, 1, 2], len(noise_idx))

# Create DataFrame
df_syn = pd.DataFrame({
    'flow_rate': flow,
    'pressure': pressure,
    'temperature': temperature,
    'power_input': power_in,
    'power_output': power_out,
    'valve_position': valve,
    'pump_speed': pump,
    'label': labels
})

df_syn.to_csv("synthetic_ics_enhanced.csv", index=False)
print(f"  ✓ Saved 'synthetic_ics_enhanced.csv'")
print(f"  ✓ Label distribution:")
for label, count in df_syn['label'].value_counts().sort_index().items():
    label_name = ['Normal', 'Anomaly', 'Fault'][label]
    print(f"     - {label_name}: {count} ({count/len(df_syn)*100:.1f}%)")

# ============================================================================
# STEP 2: ADVANCED FEATURE ENGINEERING
# ============================================================================
print("\n[STEP 2/8] Engineering Advanced Physics-Aware Features...")

df = df_syn.copy()

# Basic derived features
df['energy_residual'] = df['power_input'] - df['power_output']
df['efficiency'] = df['power_output'] / (df['power_input'] + 1)
df['temp_flow_ratio'] = df['temperature'] / (df['flow_rate'] + 1)
df['pressure_flow_ratio'] = df['pressure'] / (df['flow_rate']**2 + 1)
df['control_mismatch'] = np.abs(df['valve_position'] - df['pump_speed'])

# Rolling statistics (temporal patterns)
window_size = 10
df['flow_rate_rolling_mean'] = df['flow_rate'].rolling(window=window_size, min_periods=1).mean()
df['flow_rate_rolling_std'] = df['flow_rate'].rolling(window=window_size, min_periods=1).std()
df['temperature_rolling_mean'] = df['temperature'].rolling(window=window_size, min_periods=1).mean()
df['pressure_rolling_std'] = df['pressure'].rolling(window=window_size, min_periods=1).std()

# Rate of change features
df['flow_rate_change'] = df['flow_rate'].diff().fillna(0)
df['temperature_change'] = df['temperature'].diff().fillna(0)
df['pressure_change'] = df['pressure'].diff().fillna(0)

# Interaction features
df['valve_pump_interaction'] = df['valve_position'] * df['pump_speed']
df['pressure_temp_interaction'] = df['pressure'] * df['temperature'] / 1000

# Statistical features
df['power_ratio'] = df['power_output'] / (df['power_input'] + 1)
df['flow_pressure_product'] = df['flow_rate'] * df['pressure'] / 1000

# Handle edge cases
df.replace([np.inf, -np.inf], 0, inplace=True)
df.fillna(0, inplace=True)

print(f"  ✓ Added {len(df.columns) - 8} advanced features")
print(f"  ✓ Total features: {len(df.columns) - 1}")

# ============================================================================
# STEP 3: TRAIN/TEST SPLIT
# ============================================================================
print("\n[STEP 3/8] Splitting Data with Stratification...")

X = df.drop(columns=['label'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

print(f"  ✓ Training set: {len(X_train)} samples")
print(f"  ✓ Test set: {len(X_test)} samples")
print(f"  ✓ Train distribution: {dict(y_train.value_counts().sort_index())}")
print(f"  ✓ Test distribution: {dict(y_test.value_counts().sort_index())}")

# ============================================================================
# STEP 4: ROBUST SCALING
# ============================================================================
print("\n[STEP 4/8] Applying Robust Scaling...")

# Use RobustScaler (better for outliers)
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, "scaler.pkl")
print("  ✓ Saved 'scaler.pkl'")

# ============================================================================
# STEP 5: TRAIN ENHANCED AUTOENCODER
# ============================================================================
if HAS_TENSORFLOW:
    print("\n[STEP 5/8] Training Enhanced Autoencoder...")
    
    input_dim = X_train_scaled.shape[1]
    
    # Enhanced architecture with regularization
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(128, activation='relu')(input_layer)
    encoded = BatchNormalization()(encoded)
    encoded = Dropout(0.2)(encoded)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = BatchNormalization()(encoded)
    encoded = Dropout(0.2)(encoded)
    encoded = Dense(32, activation='relu')(encoded)
    
    decoded = Dense(64, activation='relu')(encoded)
    decoded = BatchNormalization()(decoded)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = BatchNormalization()(decoded)
    output = Dense(input_dim)(decoded)

    autoencoder = Model(inputs=input_layer, outputs=output)
    autoencoder.compile(optimizer='adam', loss='mse')

    # Enhanced callbacks
    callbacks = [
        EarlyStopping(patience=8, restore_best_weights=True, monitor='val_loss'),
        ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-6)
    ]

    history = autoencoder.fit(
        X_train_scaled, X_train_scaled,
        epochs=50,
        batch_size=128,
        validation_split=0.15,
        callbacks=callbacks,
        verbose=0
    )
    
    autoencoder.save("ics_autoencoder.h5")
    print("  ✓ Enhanced Autoencoder trained and saved")
else:
    print("\n[STEP 5/8] Skipping Autoencoder (TensorFlow not available)")

# ============================================================================
# STEP 6: TRAIN OPTIMIZED XGBOOST
# ============================================================================
print("\n[STEP 6/8] Training Optimized XGBoost with Hyperparameter Tuning...")

# Enhanced hyperparameters
xgb_model = XGBClassifier(
    n_estimators=600,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    eval_metric='mlogloss',
    random_state=42,
    n_jobs=-1,
    verbosity=0,
    scale_pos_weight=1.5  # Handle class imbalance
)

print("  → Training with optimized hyperparameters...")
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

joblib.dump(xgb_model, "ics_xgboost.pkl")
print("  ✓ Optimized XGBoost trained and saved")

# Save feature names for SHAP
joblib.dump(X_train.columns.tolist(), "feature_names.pkl")
print("  ✓ Saved feature names")

# ============================================================================
# STEP 7: COMPREHENSIVE EVALUATION
# ============================================================================
print("\n[STEP 7/8] Comprehensive Model Evaluation...")

y_pred = xgb_model.predict(X_test)
y_proba = xgb_model.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"\n  ✓ Overall Accuracy: {accuracy*100:.2f}%")
print(f"  ✓ Weighted F1-Score: {f1*100:.2f}%")

print("\n" + "="*70)
print("DETAILED CLASSIFICATION REPORT")
print("="*70)
print(classification_report(y_test, y_pred, 
                          target_names=['Normal', 'Anomaly', 'Fault'],
                          digits=4,
                          zero_division=0))

# Enhanced Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\n" + "="*70)
print("CONFUSION MATRIX")
print("="*70)
print("              Normal  Anomaly    Fault")
for i, label in enumerate(['Normal', 'Anomaly', 'Fault']):
    if i < cm.shape[0]:
        print(f"{label:12s}  {cm[i,0]:6d}  {cm[i,1]:7d}  {cm[i,2]:7d}")

# Visualize Enhanced Confusion Matrix
plt.figure(figsize=(12, 9))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
            xticklabels=['Normal', 'Anomaly', 'Fault'],
            yticklabels=['Normal', 'Anomaly', 'Fault'],
            cbar_kws={'label': 'Count'})
plt.title('Enhanced Confusion Matrix - ICS Anomaly Detection', 
          fontsize=18, fontweight='bold', pad=20)
plt.ylabel('True Label', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix_enhanced.png', dpi=300, bbox_inches='tight')
print("\n  ✓ Enhanced confusion matrix saved")
plt.close()

# Feature Importance
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False).head(15)

plt.figure(figsize=(12, 8))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Importance', fontsize=12, fontweight='bold')
plt.ylabel('Feature', fontsize=12, fontweight='bold')
plt.title('Top 15 Feature Importances (XGBoost)', fontsize=16, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("  ✓ Feature importance plot saved")
plt.close()

# ============================================================================
# STEP 8: ENHANCED SHAP ANALYSIS
# ============================================================================
if HAS_SHAP:
    print("\n[STEP 8/8] Generating Enhanced SHAP Explanations...")
    
    # Sample for SHAP
    sample_size = min(800, len(X_test))
    X_shap_sample = X_test.sample(n=sample_size, random_state=42)
    
    print("  → Computing SHAP values (this may take a few minutes)...")
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_shap_sample)
    
    # Save SHAP explainer
    joblib.dump(explainer, "shap_explainer.pkl")
    print("  ✓ SHAP explainer saved")
    
    # Global Feature Importance (Bar Plot)
    print("  → Creating global SHAP importance plots...")
    plt.figure(figsize=(14, 10))
    shap.summary_plot(shap_values, X_shap_sample, plot_type="bar", 
                     show=False, max_display=15, class_names=['Normal', 'Anomaly', 'Fault'])
    plt.title('Global Feature Importance (SHAP Values)', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig('shap_global_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Detailed Summary Plot for Anomaly Class
    if isinstance(shap_values, list) and len(shap_values) > 1:
        plt.figure(figsize=(14, 12))
        shap.summary_plot(shap_values[1], X_shap_sample, show=False, max_display=15)
        plt.title('SHAP Summary - Anomaly Class Feature Impact', fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig('shap_anomaly_detailed.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Fault Class
        if len(shap_values) > 2:
            plt.figure(figsize=(14, 12))
            shap.summary_plot(shap_values[2], X_shap_sample, show=False, max_display=15)
            plt.title('SHAP Summary - Fault Class Feature Impact', fontsize=18, fontweight='bold')
            plt.tight_layout()
            plt.savefig('shap_fault_detailed.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    print("  ✓ All SHAP visualizations saved")
    
else:
    print("\n[STEP 8/8] Skipping SHAP Analysis (library not available)")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("✅ ENHANCED TRAINING COMPLETE!")
print("="*80)
print("\n📊 Performance Summary:")
print(f"  - Dataset Size: {len(df):,} samples")
print(f"  - Total Features: {len(X.columns)}")
print(f"  - Model Accuracy: {accuracy*100:.2f}%")
print(f"  - Weighted F1-Score: {f1*100:.2f}%")
print(f"\n📁 Files Created:")
print("  ✓ synthetic_ics_enhanced.csv       - Enhanced training data")
print("  ✓ scaler.pkl                       - Robust feature scaler")
print("  ✓ ics_xgboost.pkl                  - Optimized XGBoost model")
print("  ✓ feature_names.pkl                - Feature names for SHAP")
if HAS_TENSORFLOW:
    print("  ✓ ics_autoencoder.h5               - Enhanced autoencoder")
print("  ✓ confusion_matrix_enhanced.png    - Performance visualization")
print("  ✓ feature_importance.png           - Feature importance plot")
if HAS_SHAP:
    print("  ✓ shap_explainer.pkl               - SHAP explainer object")
    print("  ✓ shap_global_importance.png       - Global SHAP analysis")
    print("  ✓ shap_anomaly_detailed.png        - Anomaly class SHAP")
    print("  ✓ shap_fault_detailed.png          - Fault class SHAP")

print("\n🚀 Next Steps:")
print("  1. Review all generated plots and metrics")
print("  2. Launch the enhanced dashboard:")
print("     streamlit run app.py")
print("  3. Test with various parameter combinations")
print("  4. Monitor real-time explanations with SHAP")
print("\n" + "="*80)
print("💡 TIP: The model now includes rolling statistics and temporal features")
print("    for better anomaly detection in time-series ICS data!")
print("="*80)