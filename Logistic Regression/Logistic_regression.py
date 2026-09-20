import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# --------------------------------------------------
# Practical 2: Logistic Regression
# Customer Churn Prediction
# --------------------------------------------------

# 1. Load the dataset
df = pd.read_csv("ecommerce_customer_churn.csv")

print("Dataset Shape:", df.shape)
print("\nFirst 5 records:")
print(df.head())

# 2. Select independent (X) and dependent (Y) variables
X = df[["Age", "Tenure_Months", "Monthly_Spend_USD", "Support_Calls", "Satisfaction_Score"]]
Y = df["Churn"]  # 0 = Retained, 1 = Churned

# 3. Split dataset into training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, 
    Y, 
    test_size=0.2, 
    random_state=42
)

# 4. Create Logistic Regression model
# Increased max_iter ensures the math converges properly with 5 different features
model = LogisticRegression(max_iter=1000)

# 5. Train the model
model.fit(X_train, Y_train)

# 6. Predict outcomes for test data
Y_pred = model.predict(X_test)

# 7. Display Results
print("\n----- Logistic Regression Results -----")
print("Accuracy Score:", round(accuracy_score(Y_test, Y_pred), 4))
print("\nClassification Report:\n", classification_report(
    Y_test, 
    Y_pred, 
    target_names=["Retained (0)", "Churned (1)"]
))


# --------------------------------------------------
# 8. Visualize the Confusion Matrix
# --------------------------------------------------

plt.figure(figsize=(7, 5))
cm = confusion_matrix(Y_test, Y_pred)

# Use Seaborn to map the matrix visually
sns.heatmap(
    cm, 
    annot=True, 
    fmt="d", 
    cmap="Blues", 
    xticklabels=["Retained", "Churned"], 
    yticklabels=["Retained", "Churned"]
)

plt.xlabel("Predicted Outcome")
plt.ylabel("Actual Outcome")
plt.title("Confusion Matrix: Customer Churn Prediction")
plt.tight_layout()
plt.show()