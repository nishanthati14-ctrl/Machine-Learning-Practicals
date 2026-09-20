import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --------------------------------------------------
# Part 1: Generate Human Iris (Eye Color) Dataset
# --------------------------------------------------

np.random.seed(42)
num_per_class = 300

# Generating biologically inspired dummy data (Melanin Level & Stroma Density)
# Blue eyes: Low melanin, low-to-moderate density
blue_features = np.random.normal(loc=[15, 30], scale=[5, 8], size=(num_per_class, 2))
# Green eyes: Moderate melanin, moderate density
green_features = np.random.normal(loc=[35, 50], scale=[7, 8], size=(num_per_class, 2))
# Hazel eyes: Moderate-high melanin, higher density
hazel_features = np.random.normal(loc=[60, 65], scale=[8, 9], size=(num_per_class, 2))
# Brown eyes: High melanin, high density
brown_features = np.random.normal(loc=[85, 80], scale=[6, 7], size=(num_per_class, 2))

# Combine features and create labels
X_data = np.vstack((blue_features, green_features, hazel_features, brown_features))
labels = (
    ["Blue"] * num_per_class + 
    ["Green"] * num_per_class + 
    ["Hazel"] * num_per_class + 
    ["Brown"] * num_per_class
)

# Create DataFrame and save to CSV
df = pd.DataFrame(X_data, columns=["Melanin_Level", "Stroma_Density"])
# Ensure no negative values exist from the random generation
df["Melanin_Level"] = np.clip(df["Melanin_Level"], 0, 100).round(1)
df["Stroma_Density"] = np.clip(df["Stroma_Density"], 0, 100).round(1)
df["Eye_Color"] = labels

df.to_csv("human_iris_eye_color.csv", index=False)
print("Dataset generated and saved as 'human_iris_eye_color.csv'!\n")


# --------------------------------------------------
# Part 2: Practical 3 - K-Nearest Neighbors (KNN)
# Human Eye Color Classification (K=5)
# --------------------------------------------------

# 1. Load the dataset
df_eyes = pd.read_csv("human_iris_eye_color.csv")

# 2. Select features (X) and target (Y)
X = df_eyes[["Melanin_Level", "Stroma_Density"]]
Y = df_eyes["Eye_Color"]

# 3. Split into training and testing data (80% train, 20% test)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, 
    Y, 
    test_size=0.2, 
    random_state=42
)

# 4. Create KNN model with K=5
knn_model = KNeighborsClassifier(n_neighbors=5)

# 5. Train the model
knn_model.fit(X_train, Y_train)

# 6. Predict outcomes for test data
Y_pred = knn_model.predict(X_test)

# 7. Display Results
print("----- KNN (K=5) Results -----")
print("Accuracy Score:", round(accuracy_score(Y_test, Y_pred), 4))
print("\nClassification Report:\n", classification_report(Y_test, Y_pred))

# 8. Predict eye color for a new person (e.g., Melanin 40, Density 55)
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    new_person = [[40, 55]]
    predicted_color = knn_model.predict(new_person)
    print(f"\nPrediction for new person [Melanin 40, Density 55]: {predicted_color[0]}")


# --------------------------------------------------
# 9. Visualize the Confusion Matrix
# --------------------------------------------------

plt.figure(figsize=(7, 5))
cm = confusion_matrix(Y_test, Y_pred, labels=["Blue", "Green", "Hazel", "Brown"])

# Plotting the matrix
sns.heatmap(
    cm, 
    annot=True, 
    fmt="d", 
    cmap="YlOrBr", # Yellow-Orange-Brown colormap fits the eye color theme
    xticklabels=["Blue", "Green", "Hazel", "Brown"], 
    yticklabels=["Blue", "Green", "Hazel", "Brown"]
)

plt.xlabel("Predicted Eye Color")
plt.ylabel("Actual Eye Color")
plt.title("Confusion Matrix: Human Eye Color (KNN, K=5)")
plt.tight_layout()
plt.show()