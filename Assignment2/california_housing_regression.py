import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from GPRegression import GPRegression

data = fetch_california_housing()

X = data.data
y = data.target

print(X.shape)   # (506, 13)
print(y.shape)   # (506,)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train = X_train[:100]
y_train = y_train[:100]

X_test = X_test[:20]
y_test = y_test[:20]

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
