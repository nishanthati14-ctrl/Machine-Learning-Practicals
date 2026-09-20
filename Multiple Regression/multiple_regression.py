import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Dataset
data = {
    "Area": [1000, 1200, 1500, 1800, 2000,
             2200, 2400, 2600, 2800, 3000],

    "Bedrooms": [2, 2, 3, 3, 4,
                 4, 4, 4, 5, 5],

    "Bathrooms": [1, 2, 2, 2, 3,
                  3, 3, 3, 4, 4],

    "Age": [15, 12, 10, 8, 6,
            5, 4, 3, 2, 1],

    "Price": [45, 52, 65, 75, 90,
              100, 112, 123, 140, 155]
}

# Create DataFrame
df = pd.DataFrame(data)

# Independent variables
X = df[["Area", "Bedrooms", "Bathrooms", "Age"]]

# Dependent variable
Y = df["Price"]

# Create Multiple Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X, Y)

# Predict existing data
Y_pred = model.predict(X)

# New house details
new_house = [[2200, 4, 3, 4]]

# Predict house price
prediction = model.predict(new_house)

# Evaluation
mse = mean_squared_error(Y, Y_pred)
r2 = r2_score(Y, Y_pred)

# Display results
print("Predicted House Price:",
      round(prediction[0], 2), "Lakhs")

print("Mean Squared Error:",
      round(mse, 2))

print("R2 Score:",
      round(r2, 2))