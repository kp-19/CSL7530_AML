import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from GPRegression import GPRegression

data = fetch_california_housing()

X = data.data
y = data.target

print(X.shape)   
print(y.shape)   

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train = X_train[:1000]
y_train = y_train[:1000]

X_test = X_test[:200]
y_test = y_test[:200]

mean, var, ll = GPRegression(
    X_train,
    y_train,
    X_test,
    l=1.0,
    sigma_f=1.0,
    sigma_noise=0.1
)

print("Mean function:\n ", mean)
print("Variance:\n ", var)
print("log likelihood:\n ", ll)

# Metrics and plots:

mean = mean.flatten()
var_diag = np.diag(var)

rmse = np.sqrt(mean_squared_error(y_test, mean))
mse = mean_squared_error(y_test, mean)
r2 = r2_score(y_test, mean)

print("\nRegression Metrics:")
print("     MSE :", mse)
print("     RMSE:", rmse)
print("     R2  :", r2)

# Predicted vs True plot
plt.figure(figsize=(6,6))

plt.scatter(y_test, mean, alpha=0.7)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r--')

plt.xlabel("True Prices")
plt.ylabel("Predicted Prices")
plt.title("Predicted vs True Values")

plt.show()

# Predictive var vs Absolute error
abs_error = np.abs(y_test - mean)

plt.figure(figsize=(6,5))

plt.scatter(var_diag, abs_error, alpha=0.7)

plt.xlabel("Predictive Variance")
plt.ylabel("Absolute Error")
plt.title("Error vs Predictive Variance")

plt.show()