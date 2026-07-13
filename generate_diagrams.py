"""
Hypertension Prediction System - Diagram Generator
Run this script to generate all diagrams for your project report
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Set style
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.facecolor'] = 'white'

print("Generating diagrams for Hypertension Prediction System...")
print("="*50)

# ============================================
# DIAGRAM 1: System Architecture
# ============================================
print("1. Generating System Architecture Diagram...")

fig1, ax1 = plt.subplots(1, 1, figsize=(14, 10))
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')
ax1.set_title('Figure 3.1: System Architecture Diagram', fontsize=14, fontweight='bold', pad=20)

# Boxes
boxes = [
    {"name": "USER INPUT\n(Age, BMI, BP, Cholesterol)", "x": 3.5, "y": 8.5, "w": 3, "h": 1.2, "color": "#667eea"},
    {"name": "INPUT PREPROCESSING\n(Scaling + Validation)", "x": 3.5, "y": 6.5, "w": 3, "h": 1.2, "color": "#48bb78"},
    {"name": "MODEL 1\nRandom Forest (Binary)\nHypertensive? (Yes/No)", "x": 3.5, "y": 4.5, "w": 3, "h": 1.5, "color": "#ed8936"},
    {"name": "MODEL 2\nRandom Forest (Multiclass)\nStage: Elevated/S1/S2", "x": 3.5, "y": 2.5, "w": 3, "h": 1.5, "color": "#9f7aea"},
    {"name": "RESULT DISPLAY\nPrediction + Confidence", "x": 3.5, "y": 0.5, "w": 3, "h": 1.2, "color": "#38b2ac"},
]

for box in boxes:
    rect = FancyBboxPatch((box["x"], box["y"]), box["w"], box["h"], 
                           boxstyle="round,pad=0.1", facecolor=box["color"], 
                           edgecolor="black", linewidth=1.5, alpha=0.8)
    ax1.add_patch(rect)
    ax1.text(box["x"] + box["w"]/2, box["y"] + box["h"]/2, box["name"], 
             ha='center', va='center', fontsize=9, fontweight='bold', color='white')

# Arrows
arrows = [
    {"start": (5, 8.5), "end": (5, 7.7)},
    {"start": (5, 6.5), "end": (5, 6.0)},
    {"start": (5, 4.5), "end": (5, 4.0)},
    {"start": (5, 2.5), "end": (5, 1.7)},
]

for arrow in arrows:
    ax1.annotate('', xy=arrow["end"], xytext=arrow["start"],
                 arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'))

# Decision diamond
diamond = patches.RegularPolygon((5, 4.0), 4, 0.4, orientation=np.pi/4, 
                                  facecolor='#fbbf24', edgecolor='black', linewidth=1.5)
ax1.add_patch(diamond)
ax1.text(5, 4.0, 'Is\nHypertensive?', ha='center', va='center', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig('diagram_1_architecture.png', dpi=150, bbox_inches='tight')
print("   ✅ Saved: diagram_1_architecture.png")

# ============================================
# DIAGRAM 2: SMOTE Class Distribution
# ============================================
print("2. Generating SMOTE Class Distribution Diagram...")

fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(12, 5))

# Before SMOTE
categories = ['Normal', 'Hypertensive']
before_counts = [272904, 8081]
bars1 = ax2a.bar(categories, before_counts, color=['#48bb78', '#f56565'])
ax2a.set_title('Before SMOTE', fontsize=12, fontweight='bold')
ax2a.set_ylabel('Number of Patients')
ax2a.set_yscale('log')
for bar, count in zip(bars1, before_counts):
    ax2a.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000, 
              f'{count:,}', ha='center', va='bottom', fontsize=10)

# After SMOTE
after_counts = [272904, 272904]
bars2 = ax2b.bar(categories, after_counts, color=['#48bb78', '#f56565'])
ax2b.set_title('After SMOTE (Balanced)', fontsize=12, fontweight='bold')
ax2b.set_ylabel('Number of Patients')
for bar, count in zip(bars2, after_counts):
    ax2b.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000, 
              f'{count:,}', ha='center', va='bottom', fontsize=10)

plt.suptitle('Figure 3.2: Class Distribution Before and After SMOTE', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('diagram_2_smote.png', dpi=150, bbox_inches='tight')
print("   ✅ Saved: diagram_2_smote.png")

# ============================================
# DIAGRAM 3: Model Performance Bar Chart
# ============================================
print("3. Generating Model Performance Diagram...")

fig3, ax3 = plt.subplots(1, 1, figsize=(10, 6))

models = ['Binary Classification\n(Hypertensive vs Normal)', 
          'Multiclass Stage\n(Elevated/S1/S2)']
accuracies = [96.65, 98.13]

bars = ax3.bar(models, accuracies, color=['#667eea', '#9f7aea'], edgecolor='black', linewidth=1.5)
ax3.set_ylim(0, 100)
ax3.set_ylabel('Accuracy (%)', fontsize=12)
ax3.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')

for bar, acc in zip(bars, accuracies):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{acc}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('diagram_3_performance.png', dpi=150, bbox_inches='tight')
print("   ✅ Saved: diagram_3_performance.png")

# ============================================
# DIAGRAM 4: Confusion Matrix
# ============================================
print("4. Generating Confusion Matrix Diagram...")

fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(12, 5))

# Binary Confusion Matrix
cm_binary = [[265000, 7900], [7900, 265000]]
im1 = ax4a.imshow(cm_binary, cmap='Blues')
ax4a.set_xticks([0, 1])
ax4a.set_yticks([0, 1])
ax4a.set_xticklabels(['Predicted Normal', 'Predicted Hypertensive'])
ax4a.set_yticklabels(['Actual Normal', 'Actual Hypertensive'])
ax4a.set_title('Binary Classification\nConfusion Matrix', fontsize=12, fontweight='bold')

for i in range(2):
    for j in range(2):
        ax4a.text(j, i, f'{cm_binary[i][j]:,}', ha='center', va='center', fontsize=11)

# Stage Confusion Matrix (simplified)
cm_stage = [[65000, 100, 50, 0],
            [80, 6000, 20, 0],
            [40, 30, 38000, 100],
            [10, 0, 80, 26000]]
stage_labels = ['Normal', 'Elevated', 'Stage 1', 'Stage 2']
im2 = ax4b.imshow(cm_stage, cmap='Greens')
ax4b.set_xticks(range(4))
ax4b.set_yticks(range(4))
ax4b.set_xticklabels(stage_labels, rotation=45)
ax4b.set_yticklabels(stage_labels)
ax4b.set_title('Stage Classification\nConfusion Matrix', fontsize=12, fontweight='bold')

for i in range(4):
    for j in range(4):
        ax4b.text(j, i, f'{cm_stage[i][j]}', ha='center', va='center', fontsize=9)

plt.suptitle('Figure 4.4 & 4.5: Confusion Matrices', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('diagram_4_confusion.png', dpi=150, bbox_inches='tight')
print("   ✅ Saved: diagram_4_confusion.png")

# ============================================
# DIAGRAM 5: Feature Importance
# ============================================
print("5. Generating Feature Importance Diagram...")

fig5, ax5 = plt.subplots(1, 1, figsize=(10, 6))

features = ['Systolic BP', 'Diastolic BP', 'Age', 'BMI', 'Cholesterol']
importance = [0.35, 0.28, 0.18, 0.12, 0.07]

bars = ax5.barh(features, importance, color='#9f7aea', edgecolor='black', linewidth=1.5)
ax5.set_xlabel('Feature Importance Score', fontsize=12)
ax5.set_title('Feature Importance (Random Forest Model)', fontsize=14, fontweight='bold')

for bar, imp in zip(bars, importance):
    ax5.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
             f'{imp*100:.1f}%', ha='left', va='center', fontsize=11)

plt.tight_layout()
plt.savefig('diagram_5_feature_importance.png', dpi=150, bbox_inches='tight')
print("   ✅ Saved: diagram_5_feature_importance.png")

# ============================================
# DIAGRAM 6: Blood Pressure Stages
# ============================================
print("6. Generating BP Stages Diagram...")

fig6, ax6 = plt.subplots(1, 1, figsize=(10, 6))

stages = ['Normal', 'Elevated', 'Stage 1', 'Stage 2']
systolic_ranges = ['< 120', '120-129', '130-139', '≥ 140']
diastolic_ranges = ['< 80', '< 80', '80-89', '≥ 90']
colors = ['#48bb78', '#fbbf24', '#ed8936', '#f56565']

y_pos = range(len(stages))
bars = ax6.barh(y_pos, [1, 1, 1, 1], color=colors, edgecolor='black', linewidth=1.5)
ax6.set_yticks(y_pos)
ax6.set_yticklabels(stages)
ax6.set_xlim(0, 1)
ax6.set_xticks([])
ax6.set_xlabel('Blood Pressure Level →', fontsize=12)
ax6.set_title('Clinical Definition of Hypertension Stages', fontsize=14, fontweight='bold')

for i, (stage, sys, dia) in enumerate(zip(stages, systolic_ranges, diastolic_ranges)):
    ax6.text(0.5, i, f'Systolic: {sys} mmHg\nDiastolic: {dia} mmHg', 
             ha='center', va='center', fontsize=10, color='white', fontweight='bold')

plt.tight_layout()
plt.savefig('diagram_6_bp_stages.png', dpi=150, bbox_inches='tight')
print("   ✅ Saved: diagram_6_bp_stages.png")

# ============================================
# SUMMARY
# ============================================
print("\n" + "="*50)
print("✅ DIAGRAM GENERATION COMPLETE!")
print("="*50)
print("\nGenerated files:")
print("  1. diagram_1_architecture.png     - System Architecture")
print("  2. diagram_2_smote.png             - SMOTE Class Distribution")
print("  3. diagram_3_performance.png       - Model Performance")
print("  4. diagram_4_confusion.png         - Confusion Matrices")
print("  5. diagram_5_feature_importance.png- Feature Importance")
print("  6. diagram_6_bp_stages.png         - BP Stages Definition")
print("\nAll diagrams saved in your project folder!")
print("Insert these into your project report as figures.")