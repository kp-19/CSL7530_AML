import numpy as np
from sklearn.model_selection import train_test_split
from GradientBoostingRegression import GradientBoostingRegression, inference_regression
from sklearn.datasets import fetch_california_housing

# Load dataset
data = fetch_california_housing()

X = data.data
y = data.target

print("Loaded california dataset successfully")
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

print("features :",data.feature_names)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

f0, trees, lr, loss_history = GradientBoostingRegression(X_train, y_train, M=100, max_depth=3, min_samples_split=10, min_samples_leaf=5, lr=0.05, save_loss_interval=1)

y_pred_train = inference_regression(X_train, f0, trees, lr)
y_pred_test = inference_regression(X_test, f0, trees, lr)

train_mse = np.mean((y_train - y_pred_train) ** 2)
test_mse = np.mean((y_test - y_pred_test) ** 2)

print("Train MSE:", train_mse)
print("Test MSE:", test_mse)

