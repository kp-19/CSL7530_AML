import numpy as np
import matplotlib.pyplot as plt
from GPRegression import GPRegression

np.random.seed(42)

N = 30

X_train = np.random.uniform(-3, 3, (N, 2))
y_train = np.sin(X_train[:,0]**2 + X_train[:,1]**2)
sigma_noise = 0.1
y_train = y_train + np.random.normal(0, sigma_noise, N)

# print("X_train:", X_train)
# print("y_train:", y_train)

grid_n = 30

x1 = np.linspace(-3,3,grid_n)
x2 = np.linspace(-3,3,grid_n)
X1, X2 = np.meshgrid(x1, x2)
X_test = np.column_stack([X1.ravel(), X2.ravel()])

# print("X_test:", X_test)

mean, var, _ = GPRegression(X_train, y_train, X_test)
mean = mean.reshape(grid_n, grid_n)
var = np.diag(var)
var = var.reshape(grid_n, grid_n)

# Predictive Mean plot
plt.figure(figsize=(7,6))

plt.contourf(X1, X2, mean, 30)
plt.colorbar(label="Predictive Mean")

plt.scatter(X_train[:,0], X_train[:,1], c='red', label="Training points")

plt.title("GP Predictive Mean")
plt.legend()

plt.show()

# Predictive Variance
plt.figure(figsize=(7,6))

plt.contourf(X1, X2, var, 30)
plt.colorbar(label="Predictive Variance")

plt.scatter(X_train[:,0], X_train[:,1], c='red')

plt.title("GP Predictive Variance")

plt.show()

# Covariance Matrix for a particular test input:
x_star = np.array([[0.5, -1.0]])

mean, cov, _ = GPRegression(X_train, y_train, x_star)

print("Predictive mean:", mean)
print("Predictive variance:", cov)

# True function vs GP prediction:
true_surface = np.sin(X1**2 + X2**2)

plt.figure(figsize=(7,6))

plt.contourf(X1, X2, true_surface, 30)
plt.colorbar()

plt.scatter(X_train[:,0], X_train[:,1], c='red')

plt.title("True Function Surface")

plt.show()