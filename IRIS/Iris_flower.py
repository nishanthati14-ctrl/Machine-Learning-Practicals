import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --------------------------------------------------
# Backup Practical: K-Nearest Neighbors (KNN)
# Traditional Iris Flower Classification (K=5)
# --------------------------------------------------

# 1. Load the built-in Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
Y = iris.target  # Classes: 0 (setosa), 1 (versicolor), 2 (virginica)

print("Dataset Shape:", X.shape)
print("\nFirst 5 records:")
print(X.head())

# 2. Split dataset into training and testing data (80% train, 20% test)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, 
    Y, 
    test_size=0.2, 
    random_state=42
)

# 3. Create KNN model and set K=5
knn_model = KNeighborsClassifier(n_neighbors=5)

# 4. Train the model
knn_model.fit(X_train, Y_train)

# 5. Predict outcomes for the test data
Y_pred = knn_model.predict(X_test)

# 6. Display evaluation metrics
print("\n----- KNN (K=5) Results -----")
print("Accuracy Score:", round(accuracy_score(Y_test, Y_pred), 4))
print("\nClassification Report:\n", classification_report(
    Y_test, 
    Y_pred, 
    target_names=iris.target_names
))

# 7. Visualize with a Confusion Matrix
plt.figure(figsize=(7, 5))
cm = confusion_matrix(Y_test, Y_pred)

# Using a Green color map since it's a plant dataset
sns.heatmap(
    cm, 
    annot=True, 
    fmt="d", 
    cmap="Greens", 
    xticklabels=iris.target_names, 
    yticklabels=iris.target_names
)

plt.xlabel("Predicted Species")
plt.ylabel("Actual Species")
plt.title("Confusion Matrix: Iris Flower Classification (KNN, K=5)")
plt.tight_layout()
plt.show()