import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from GradientBoostingClassification import GradientBoostingClassifier, inference_classification

data = load_digits()

X = data.data      # shape (1797, 64)
y = data.target    # shape (1797,)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Classes:", np.unique(y))

X = X / 16.0

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

trees, loss_history = GradientBoostingClassifier(
    X_train,
    y_train,
    M=20,
    max_depth=2,
    min_samples_split=5,
    min_samples_leaf=2,
    lr=0.1,
)

y_pred = inference_classification(
    X_test,
    trees,
    lr=0.1
)

acc = accuracy_score(y_test, y_pred)
print("Test Accuracy:", acc)