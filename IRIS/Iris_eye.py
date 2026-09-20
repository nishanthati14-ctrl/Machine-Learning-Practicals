import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# ---------------------------------------------------------
# 1. Load, Resize, and Flatten Images
# ---------------------------------------------------------
# The 'r' ensures Windows reads the backslashes correctly
dataset_path = r"C:\Users\MY PC\OneDrive\Desktop\Msc Data Science\IRIS\train" 

X = []
y = []

# Map folder names to numerical classes (e.g., 'eye' = 0, 'noeye' = 1)
class_names = [name for name in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, name))]
class_mapping = {class_name: idx for idx, class_name in enumerate(class_names)}

print(f"Classes detected: {class_mapping}")

for class_name in class_names:
    class_path = os.path.join(dataset_path, class_name)
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        
        # Read image in grayscale to reduce complexity
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is not None:
            # Resize to 64x64 and flatten the grid into a 1D array of 4,096 features
            img_resized = cv2.resize(img, (64, 64))
            X.append(img_resized.flatten())
            y.append(class_mapping[class_name])

X = np.array(X)
y = np.array(y)
print(f"Dataset loaded. Total images: {len(X)}, Features per image: {X.shape[1]}")

# ---------------------------------------------------------
# 2. Split and Scale the Data
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------
# 3. Models & Evaluation
# ---------------------------------------------------------

# Model 1: Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)
lin_raw_preds = lin_reg.predict(X_test_scaled)
lin_class_preds = np.where(lin_raw_preds >= 0.5, 1, 0) # Threshold continuous values to 0 or 1

print("\n--- Linear Regression ---")
print(f"Accuracy: {accuracy_score(y_test, lin_class_preds):.4f}")

# Model 2: Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train)
log_preds = log_reg.predict(X_test_scaled)

print("\n--- Logistic Regression ---")
print(f"Accuracy: {accuracy_score(y_test, log_preds):.4f}")

# Model 3: K-Nearest Neighbors (K=5)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
knn_preds = knn.predict(X_test_scaled)

print("\n--- K-Nearest Neighbors (K=5) ---")
print(f"Accuracy: {accuracy_score(y_test, knn_preds):.4f}")


# ---------------------------------------------------------
# 4. Visual Representations
# ---------------------------------------------------------

# Confusion Matrices
def plot_cm(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=list(class_mapping.keys()), 
                yticklabels=list(class_mapping.keys()))
    plt.title(title)
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

plot_cm(y_test, lin_class_preds, 'Linear Regression Confusion Matrix')
plot_cm(y_test, log_preds, 'Logistic Regression Confusion Matrix')
plot_cm(y_test, knn_preds, 'KNN (K=5) Confusion Matrix')

# Linear Regression Threshold Graph
plt.figure(figsize=(8, 5))
plt.scatter(range(len(y_test)), lin_raw_preds, c=y_test, cmap='coolwarm', alpha=0.8, edgecolors='k')
plt.axhline(y=0.5, color='black', linestyle='--', linewidth=2, label='0.5 Decision Boundary')
plt.title('Linear Regression: Raw Predictions vs. Decision Threshold')
plt.xlabel('Test Image Index')
plt.ylabel('Raw Predicted Value')
plt.legend()
plt.tight_layout()
plt.show()