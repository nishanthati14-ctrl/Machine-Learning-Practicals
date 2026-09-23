import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load Dataset
# Ensure 'heart.csv' is downloaded from Kaggle and placed in the same folder as this script
try:
    df = pd.read_csv('heart.csv')
    print("Dataset successfully loaded. Shape:", df.shape)
except FileNotFoundError:
    print("Error: 'heart.csv' not found. Please download it and place it in this directory.")
    exit()

# The target column in the Kaggle Heart Disease dataset is named 'target'
X = df.drop('target', axis=1)
y = df['target']

# 2. Split Data (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Feature Scaling
# While Naive Bayes doesn't strictly require scaling, it ensures comparison parameters 
# (like resting blood pressure vs cholesterol) are weighted evenly.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Apply Naive Bayes Model
nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train)

# 5. Predict Resultant
y_pred = nb_model.predict(X_test_scaled)

# ---------------------------------------------------------
# 6. Evaluation & Comparison Parameters
# ---------------------------------------------------------
print("\n" + "="*50)
print("--- Actual vs. Predicted Resultants (First 15 Patients) ---")
print("="*50)
comparison_df = pd.DataFrame({
    'Actual Resultant (y_test)': y_test, 
    'Predicted Resultant (y_pred)': y_pred
})
# Resetting index for a cleaner display
print(comparison_df.head(15).to_string(index=False))

print("\n" + "="*50)
print("--- Model Performance Parameters ---")
print("="*50)
print(f"Accuracy Score: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['No Disease (0)', 'Disease (1)']))

# ---------------------------------------------------------
# 7. Visual Representation (Confusion Matrix)
# ---------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=['No Disease', 'Disease'], 
            yticklabels=['No Disease', 'Disease'])
plt.title('Naive Bayes Confusion Matrix (Heart Disease)')
plt.ylabel('Actual Resultant')
plt.xlabel('Predicted Resultant')
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 8. ROC Curve & AUC
# ---------------------------------------------------------
# Get the probability scores for the positive class (Disease = 1)
y_prob = nb_model.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--') # 50/50 guessing line
plt.title('ROC Curve for Heart Disease Prediction')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 9. Gaussian Feature Distribution (KDE Plot)
# ---------------------------------------------------------
# We plot the original 'thalach' (Maximum Heart Rate) feature to show 
# how the two target classes separate naturally, which Naive Bayes relies on.
plt.figure(figsize=(6, 4))
sns.kdeplot(data=df, x='thalach', hue='target', fill=True, common_norm=False, palette='coolwarm')
plt.title('Gaussian Distribution of Max Heart Rate\n(Feature: thalach vs. Target)')
plt.xlabel('Maximum Heart Rate Achieved')
plt.ylabel('Density')
# Update the legend to match the classes
plt.legend(title='Target', labels=['Disease (1)', 'No Disease (0)'])
plt.tight_layout()
plt.show()