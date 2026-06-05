import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

print("Loading hypertension-only dataset...")
df = pd.read_csv("hypertension_only.csv")

features = ['age', 'bmi', 'systolic_bp', 'diastolic_bp', 'cholesterol']

# ============================================
# MODEL 1: Binary Classification (Hypertensive or Not)
# ============================================
print("\n" + "="*50)
print("MODEL 1: Hypertension Detection")
print("="*50)

X1 = df[features].copy()
y1 = df['is_hypertensive'].copy()

print(f"Class 0 (Normal): {(y1==0).sum()}")
print(f"Class 1 (Hypertensive): {(y1==1).sum()}")

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

scaler1 = StandardScaler()
X1_train_scaled = scaler1.fit_transform(X1_train)
X1_test_scaled = scaler1.transform(X1_test)

model1 = RandomForestClassifier(n_estimators=100, random_state=42)
model1.fit(X1_train_scaled, y1_train)

y1_pred = model1.predict(X1_test_scaled)
acc1 = accuracy_score(y1_test, y1_pred)
print(f"Accuracy: {acc1*100:.2f}%")

# ============================================
# MODEL 2: Multi-class Stage Classification
# ============================================
print("\n" + "="*50)
print("MODEL 2: Stage Classification")
print("="*50)

# Filter out stages with zero samples
stage_counts = df['hypertension_stage'].value_counts()
print("Original stage distribution:")
print(stage_counts)

# Only keep stages that have samples
valid_stages = stage_counts[stage_counts > 0].index.tolist()
print(f"\nValid stages (with samples): {valid_stages}")

# Filter data to only valid stages
df_stages = df[df['hypertension_stage'].isin(valid_stages)].copy()

# Create encoded labels
label_encoder = LabelEncoder()
df_stages['stage_encoded'] = label_encoder.fit_transform(df_stages['hypertension_stage'])

X2 = df_stages[features].copy()
y2 = df_stages['stage_encoded'].copy()

print("\nStage distribution after filtering:")
for stage in valid_stages:
    count = (df_stages['hypertension_stage'] == stage).sum()
    print(f"  {stage}: {count}")

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

scaler2 = StandardScaler()
X2_train_scaled = scaler2.fit_transform(X2_train)
X2_test_scaled = scaler2.transform(X2_test)

model2 = RandomForestClassifier(n_estimators=100, random_state=42)
model2.fit(X2_train_scaled, y2_train)

y2_pred = model2.predict(X2_test_scaled)
acc2 = accuracy_score(y2_test, y2_pred)
print(f"\nStage Classification Accuracy: {acc2*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y2_test, y2_pred, target_names=label_encoder.classes_))

# ============================================
# SAVE MODELS
# ============================================
print("\nSaving models...")
joblib.dump(model1, 'model1_binary.pkl')
joblib.dump(model2, 'model2_stage.pkl')
joblib.dump(scaler1, 'scaler1.pkl')
joblib.dump(scaler2, 'scaler2.pkl')
joblib.dump(features, 'features.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

# Save stage display names
stage_display = {stage: stage.replace('_', ' ').title() for stage in valid_stages}
joblib.dump(stage_display, 'stage_display.pkl')

print("All models saved successfully!")

# ============================================
# TEST WITH EXAMPLES
# ============================================
print("\n" + "="*50)
print("TESTING WITH EXAMPLE PATIENTS")
print("="*50)

test_cases = [
    {"name": "Normal", "age": 30, "bmi": 22, "sys": 110, "dia": 70, "chol": 160},
    {"name": "Elevated", "age": 45, "bmi": 26, "sys": 125, "dia": 78, "chol": 190},
    {"name": "Stage 1", "age": 55, "bmi": 29, "sys": 135, "dia": 85, "chol": 210},
    {"name": "Stage 2", "age": 65, "bmi": 32, "sys": 155, "dia": 95, "chol": 240},
]

for case in test_cases:
    input_data = np.array([[case['age'], case['bmi'], case['sys'], case['dia'], case['chol']]])
    input_scaled = scaler1.transform(input_data)
    
    prob = model1.predict_proba(input_scaled)[0]
    pred = model1.predict(input_scaled)[0]
    
    if pred == 1:
        input_scaled2 = scaler2.transform(input_data)
        stage_pred = model2.predict(input_scaled2)[0]
        stage_name = label_encoder.inverse_transform([stage_pred])[0]
        print(f"{case['name']}: HYPERTENSIVE - Stage: {stage_name}")
    else:
        print(f"{case['name']}: NORMAL")