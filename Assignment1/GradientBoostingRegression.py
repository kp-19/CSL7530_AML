import numpy as np
from RegressionTree import RegressionTree

def GradientBoostingRegression(X, y, M=50, max_depth=3, min_samples_split=2, min_samples_leaf=1, lr=0.01, save_loss_interval=2):

    f0 = np.mean(y)         # initial model
    N = X.shape[0]          # no of training samples
    F = np.full(N, f0)      # predictions vector
    trees = []              # list of trees
    loss_history = []

    for m in range(M):
        # pseudo-residuals:
        r = y-F

        tree = RegressionTree(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
        tree.fit(X, r)

        F = F + lr*tree.predict(X)

        trees.append(tree)

        # Save training loss
        if (m + 1) % save_loss_interval == 0:
            mse = np.mean((y - F) ** 2)
            loss_history.append(mse)

    return f0, trees, lr, loss_history

def inference_regression(X, f0, trees, lr):

    N = X.shape[0]          
    F = np.full(N, f0)      
    for tree in trees:
        F += lr*tree.predict(X)
    return F

    
