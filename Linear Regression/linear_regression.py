import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# --------------------------------------------------
# Part 1: Generate Correlated Dataset (Massive Size)
# --------------------------------------------------

# Set random seed for reproducibility
np.random.seed(42)

# Generate 20,000 random house sizes between 500 and 5000 sq.ft
num_samples = 20000
size_sqft = np.random.randint(500, 5000, num_samples)

# Generate prices with a positive correlation
noise = np.random.normal(0, 3000000, num_samples)
price = 1000000 + (size_sqft * 8000) + noise
price = np.abs(price) # Ensure no negative prices

# Save to CSV
dummy_df = pd.DataFrame({
    "size_sqft": size_sqft,
    "price": price
})
dummy_df.to_csv("lt_house_reality_data_new.csv", index=False)
print("Dataset generated and saved as 'lt_house_reality_data_new.csv'!\n")


# --------------------------------------------------
# Part 2: Practical 1 - Simple Linear Regression
# House Price Prediction
# --------------------------------------------------

# 1. Load the dataset
df = pd.read_csv("lt_house_reality_data_new.csv")

# Display first 5 records
print("First 5 records:")
print(df.head())

# Display dataset shape
print("\nDataset Shape:")
print(df.shape)

# 2. Select independent and dependent variables
X = df[["size_sqft"]]
Y = df["price"]

# 3. Split dataset into training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

# 4. Create Linear Regression model
model = LinearRegression()

# 5. Train the model
model.fit(X_train, Y_train)

# 6. Predict house prices for test data
Y_pred = model.predict(X_test)

# 7. Calculate evaluation metrics
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

# 8. Predict price for a new house
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    new_house = [[1500]]
    predicted_price = model.predict(new_house)

# 9. Display results
print("\n----- Linear Regression Results -----")
print("Coefficient:", round(model.coef_[0], 2))
print("Intercept:", round(model.intercept_, 2))
print("Mean Squared Error:", round(mse, 2))
print("R2 Score:", round(r2, 2))
print("Predicted Price for 1500 sq.ft:", round(predicted_price[0], 2))


# 10. Visualize the results

plot_data = X_test.copy()
plot_data["Actual"] = Y_test
plot_data["Predicted"] = Y_pred

# Using the entire test set (now 4,000 points)
plot_data = plot_data.sort_values("size_sqft")

# Plot actual data with adjusted size and transparency for dense plotting
plt.scatter(
    plot_data["size_sqft"],
    plot_data["Actual"],
    label="Actual Data",
    alpha=0.3,  # Lowered transparency
    s=15        # Smaller dot size
)

# Plot regression line
plt.plot(
    plot_data["size_sqft"],
    plot_data["Predicted"],
    label="Regression Line",
    color="red",
    linewidth=2
)

plt.xlabel("House Size (sq.ft)")
plt.ylabel("House Price")
plt.title("Linear Regression: House Size vs Price")
plt.legend()
plt.show()