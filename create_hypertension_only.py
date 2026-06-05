import pandas as pd
import numpy as np
from sklearn.utils import resample

print("Loading original data...")
df = pd.read_csv("hypertension_data.csv")

# Select numeric columns
numeric_cols = ['age', 'bmi', 'systolic_bp', 'diastolic_bp', 'cholesterol']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Remove rows with missing values
df_clean = df[numeric_cols].copy()
df_clean = df_clean.dropna()
print(f"Clean data: {len(df_clean)} rows")

# Define hypertension stages based on BP readings (INCLUDING CRISIS)
def get_hypertension_stage(systolic, diastolic):
    if pd.isna(systolic) or pd.isna(diastolic):
        return 'UNKNOWN'
    
    # Crisis is highest priority
    if systolic >= 180 or diastolic >= 120:
        return 'CRISIS'
    elif systolic >= 140 or diastolic >= 90:
        return 'STAGE_2'
    elif systolic >= 130 or diastolic >= 80:
        return 'STAGE_1'
    elif systolic >= 120:
        return 'ELEVATED'
    else:
        return 'NORMAL'

# Apply stage classification
df_clean['hypertension_stage'] = df_clean.apply(
    lambda row: get_hypertension_stage(row['systolic_bp'], row['diastolic_bp']), 
    axis=1
)

# Create binary target (hypertensive or not) - everything except NORMAL is hypertensive
df_clean['is_hypertensive'] = (df_clean['hypertension_stage'] != 'NORMAL').astype(int)

print("\nStage distribution (with CRISIS added):")
print(df_clean['hypertension_stage'].value_counts())

# Create balanced dataset for binary classification
print("\nCreating balanced dataset...")
normal = df_clean[df_clean['is_hypertensive'] == 0]
hypertensive = df_clean[df_clean['is_hypertensive'] == 1]

print(f"Normal: {len(normal)}, Hypertensive: {len(hypertensive)}")

# Upsample to balance
if len(hypertensive) < len(normal):
    hypertensive_upsampled = resample(hypertensive, replace=True, n_samples=len(normal), random_state=42)
    balanced = pd.concat([normal, hypertensive_upsampled])
else:
    normal_upsampled = resample(normal, replace=True, n_samples=len(hypertensive), random_state=42)
    balanced = pd.concat([normal_upsampled, hypertensive])

print(f"Balanced dataset: {len(balanced)} rows")

# Save
balanced.to_csv('hypertension_only.csv', index=False)
print("\nSaved to hypertension_only.csv")

# Show final stage distribution in balanced dataset
print("\nFinal stage distribution in balanced dataset:")
print(balanced['hypertension_stage'].value_counts())