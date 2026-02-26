import numpy as np
import math

class TreeNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None
    

class RegressionTree:
    def __init__(self, max_depth=3, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        if depth>=self.max_depth or X.shape[0] < self.min_samples_split or np.var(y) < 1e-10:
            return TreeNode(value=np.mean(y))
        
        feature, threshold = self._best_split(X, y)
        if feature is None:
            return TreeNode(value=np.mean(y))

        left_mask = X[:, feature] <= threshold
        right_mask = X[:, feature] > threshold

        if left_mask.sum() == 0 or right_mask.sum() == 0:
            return TreeNode(value=np.mean(y))

        left_child = self._build_tree(X[left_mask], y[left_mask], depth+1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth+1)

        return TreeNode(feature_index=feature, threshold=threshold, left=left_child, right=right_child)

    def _calculate_tse(self, y):        # total squared error
        if len(y) == 0:
            return 0
        mean = np.mean(y)
        return np.sum((y - mean) ** 2)

    def _best_split(self, X, y):
        n_samples, n_features = X.shape

        if n_samples < self.min_samples_split:
            return None, None
        
        best_feature = None
        best_threshold = None
        best_loss = float("inf")

        for feature in range(n_features):

            sorted_indices = np.argsort(X[:, feature])
            X_sorted = X[sorted_indices]
            y_sorted = y[sorted_indices]

            for i in range(1, n_samples):
                
                # Skip identical feature values (no real split)
                if X_sorted[i, feature] == X_sorted[i-1, feature]:
                    continue
                
                # check for min_split_leaf value
                if i<self.min_samples_leaf or (n_samples-i) < self.min_samples_leaf:
                    continue

                threshold = (X_sorted[i, feature] + X_sorted[i-1, feature]) / 2

                y_left = y_sorted[:i]
                y_right = y_sorted[i:]

                # Calculate loss
                left_loss = self._calculate_tse(y_left)
                right_loss = self._calculate_tse(y_right)               

                total_loss = left_loss + right_loss

                if total_loss < best_loss:
                    best_loss = total_loss
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def predict(self, X):
        return np.array([self._predict_sample(x, self.root) for x in X])
    
    def _predict_sample(self, x, node):
        if node.is_leaf():
            return node.value
        
        if x[node.feature_index] <= node.threshold:
            return self._predict_sample(x, node.left)
        
        else:
            return self._predict_sample(x, node.right)
        

# if __name__ == "__main__":
#     import matplotlib.pyplot as plt

#     # Generate synthetic regression dataset
#     np.random.seed(42)

#     X = np.linspace(0, 10, 200).reshape(-1, 1)
#     y = X.flatten()**2 + np.random.normal(0, 5, size=200)

#     # Train tree
#     tree = RegressionTree(max_depth=6, min_samples_split=5, min_samples_leaf=2)
#     tree.fit(X, y)

#     # Predictions
#     y_pred = tree.predict(X)

#     # Plot results
#     plt.figure(figsize=(8, 5))
#     plt.scatter(X, y, color="lightgray", label="Data")
#     plt.plot(X, y_pred, color="red", linewidth=2, label="Tree Prediction")
#     plt.title("Regression Tree Test")
#     plt.legend()
#     plt.show()