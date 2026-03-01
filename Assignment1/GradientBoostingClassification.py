import numpy as np
from RegressionTree import RegressionTree

def convert_one_hot(y):
    # Convert y to one-hot encodings:
    k = len(np.unique(y))
    n_samples = y.shape[0]
    y_one_hot = np.zeros((n_samples, k))
    for i in range(n_samples):
        y_one_hot[i][y[i]] = 1 
    return y_one_hot

def GradientBoostingClassifier(X, y, M=50, max_depth=3, min_samples_split=2, min_samples_leaf=1, lr=0.01, save_loss_interval=2):

    # Convert y to one-hot encodings:
    K = len(np.unique(y))
    n_samples = X.shape[0]
    y_one_hot = convert_one_hot(y)
    trees = []
    loss_history = []

    # initialize logits:
    F = np.zeros((n_samples, K))

    # Boosting loop:
    for m in range(M):

        F_shifted = F - np.max(F, axis=1, keepdims=True)
        exp_F = np.exp(F_shifted)
        P = exp_F / np.sum(exp_F, axis=1, keepdims=True)

        class_trees = []

        for k in range(K):
            r = y_one_hot[:, k] - P[:, k]
            tree = RegressionTree(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
            tree.fit(X, r)

            # Newton Update:
            leaf_indices = tree.get_leaf_indices(X)

            for leaf, indices in leaf_indices.items():

                r_leaf = r[indices]
                p_leaf = P[indices, k]

                num = np.sum(r_leaf)
                den = np.sum(p_leaf*(1-p_leaf))

                if den == 0:
                    gamma = 0
                else:
                    gamma = (((K-1)/K) * num/den)
                
                leaf.value = gamma 
            
            # update logits:
            F[:, k] += lr*tree.predict(X)

            class_trees.append(tree)

        trees.append(class_trees)

        # Save cross-entropy loss
        if (m + 1) % save_loss_interval == 0:

            F_shifted = F - np.max(F, axis=1, keepdims=True)
            exp_F = np.exp(F_shifted)
            P = exp_F / np.sum(exp_F, axis=1, keepdims=True)

            eps = 1e-15
            loss = -np.sum(y_one_hot * np.log(P + eps)) / n_samples

            loss_history.append(loss)

    return trees, loss_history


def inference_classification(X, trees, lr):

    n_samples = X.shape[0]
    M = len(trees)
    K = len(trees[0])

    F = np.zeros((n_samples, K))

    for m in range(M):
        for k in range(K):
            F[:, k] += lr*trees[m][k].predict(X)
        
    # softmax:
    F_shifted = F - np.max(F, axis=1, keepdims=True)
    exp_F = np.exp(F_shifted)
    P = exp_F / np.sum(exp_F, axis=1, keepdims=True)

    return np.argmax(P, axis=1)    


        




