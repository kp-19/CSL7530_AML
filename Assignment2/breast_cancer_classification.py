import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

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

print("\nModel Accuracy:", accuracy)
print("\nClassification Report")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")

plt.show()

# Histogram of predicted probabilities
plt.figure(figsize=(6,4))

plt.hist(pi_star, bins=20)

plt.xlabel("Predicted Probability (Class +1)")
plt.ylabel("Frequency")
plt.title("Distribution of Predicted Probabilities")

plt.show()