import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from GPClassification import compute_fhat, GPClassification

data = load_breast_cancer()

X = data.data
y = data.target

# convert labels to {-1,+1}
y = 2*y - 1

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

f_hat, a, ll, lm = compute_fhat(X_train, y_train)
print("Log-likelihood = ", ll)
print("Marginal-likelihood = ", lm)
print("F_hat = ", f_hat)

pi_star, f_bar, var = GPClassification(X_train, y_train, X_test, f_hat)

y_pred = np.where(pi_star >= 0.5, 1, -1)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)
