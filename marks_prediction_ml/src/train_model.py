# train_model.py
# Predict marks based on hours studied using Linear Regression

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# -------------------------
# Step 1: Load the dataset
# -------------------------
data_path = "../data/study_data.csv"
df = pd.read_csv(data_path)

# -------------------------
# Step 2: Visualize data
# -------------------------
plt.scatter(df['hours'], df['marks'])
plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.title("Hours Studied vs Marks")
plt.show()

# -------------------------
# Step 3: Split input/output
# -------------------------
X = df[['hours']]   # must be 2D
y = df['marks']     # 1D

# -------------------------
# Step 4: Train-test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# Step 5: Train the model
# -------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------
# Step 6: Model parameters
# -------------------------
print("Slope (m):", model.coef_[0])
print("Intercept (c):", model.intercept_)

# -------------------------
# Step 7: Predictions
# -------------------------
y_pred = model.predict(X_test)

# -------------------------
# Step 8: Evaluation
# -------------------------
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# -------------------------
# Step 9: Predict new value
# -------------------------
hours = [[6.5]]
predicted_marks = model.predict(hours)
print("Predicted marks for 6.5 hours:", predicted_marks[0])

# -------------------------
# Step 10: Plot regression line
# -------------------------
plt.scatter(df['hours'], df['marks'], color='blue')
plt.plot(df['hours'], model.predict(df[['hours']]), color='red')
plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.title("Linear Regression Fit")
plt.show()
