import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing, load_digits
from sklearn.model_selection import train_test_split

# Import YOUR implementation
from RegressionTree import RegressionTree


# -------------------------------------------------
# Gradient Boosting Regression (Custom Trees Only)
# -------------------------------------------------
def GradientBoostingRegression(X, y, X_test, y_test,
                               M=300,
                               max_depth=3,
                               min_samples_split=10,
                               min_samples_leaf=5,
                               lr=0.05,
                               save_loss_interval=5):

    # Initial model: mean prediction
    f0 = np.mean(y)

    N_train = X.shape[0]
    N_test = X_test.shape[0]

    F_train = np.full(N_train, f0)
    F_test = np.full(N_test, f0)

    trees = []
    train_loss_history = []
    test_loss_history = []

    for m in range(M):

        # Compute pseudo-residuals
        residuals = y - F_train

        # Train regression tree on residuals
        tree = RegressionTree(max_depth=max_depth,
                              min_samples_split=min_samples_split,
                              min_samples_leaf=min_samples_leaf)

        tree.fit(X, residuals)

        # Update predictions
        F_train += lr * tree.predict(X)
        F_test += lr * tree.predict(X_test)

        trees.append(tree)

        # Save loss every save_loss_interval
        if (m + 1) % save_loss_interval == 0:
            train_mse = np.mean((y - F_train) ** 2)
            test_mse = np.mean((y_test - F_test) ** 2)

            train_loss_history.append(train_mse)
            test_loss_history.append(test_mse)

            print(f"Iteration {m+1}: "
                  f"Train MSE = {train_mse:.4f}, "
                  f"Test MSE = {test_mse:.4f}")

    return f0, trees, lr, train_loss_history, test_loss_history


# -------------------------------------------------
# Train Model and Plot MSE vs M
# -------------------------------------------------
def train_and_plot():

    # Load California Housing dataset
    data = fetch_california_housing()
    X = data.data
    y = data.target

    # 80-20 train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train Gradient Boosting
    f0, trees, lr, train_loss, test_loss = GradientBoostingRegression(
        X_train, y_train,
        X_test, y_test,
        M=500,
        max_depth=3,
        min_samples_split=10,
        min_samples_leaf=5,
        lr=0.05,
        save_loss_interval=5
    )

    # X-axis values
    iterations = np.arange(5, 301, 5)

    # Plot MSE vs M
    plt.figure()
    plt.plot(iterations, train_loss)
    plt.plot(iterations, test_loss)

    plt.xlabel("Number of Boosting Iterations (M)")
    plt.ylabel("Mean Squared Error")
    plt.title("Train and Test MSE vs M")
    plt.legend(["Train MSE", "Test MSE"])

    # Save figure for LaTeX
    plt.savefig("M_reg.png", dpi=300)
    plt.show()



# # -------------------------------
# # Utility Functions
# # -------------------------------
# def softmax(Z):
#     Z = Z - np.max(Z, axis=1, keepdims=True)  # numerical stability
#     expZ = np.exp(Z)
#     return expZ / np.sum(expZ, axis=1, keepdims=True)


# def one_hot(y, num_classes):
#     Y = np.zeros((len(y), num_classes))
#     Y[np.arange(len(y)), y] = 1
#     return Y


# # ---------------------------------------
# # Gradient Boosting for Classification
# # ---------------------------------------
# def GradientBoostingClassification(X, y, X_test, y_test,
#                                     M=100,
#                                     max_depth=3,
#                                     min_samples_split=10,
#                                     min_samples_leaf=5,
#                                     lr=0.1,
#                                     save_interval=5):

#     num_classes = len(np.unique(y))
#     N_train = X.shape[0]
#     N_test = X_test.shape[0]

#     # One-hot encoding
#     Y = one_hot(y, num_classes)

#     # Initial model: log class priors
#     class_probs = np.bincount(y) / len(y)
#     F0 = np.log(class_probs + 1e-12)

#     F_train = np.tile(F0, (N_train, 1))
#     F_test = np.tile(F0, (N_test, 1))

#     trees = [[] for _ in range(num_classes)]

#     train_acc_history = []
#     test_acc_history = []

#     for m in range(M):

#         P = softmax(F_train)

#         for k in range(num_classes):

#             # Pseudo-residuals for class k
#             r_k = Y[:, k] - P[:, k]

#             tree = RegressionTree(max_depth=max_depth,
#                                   min_samples_split=min_samples_split,
#                                   min_samples_leaf=min_samples_leaf)

#             tree.fit(X, r_k)

#             # Update predictions
#             F_train[:, k] += lr * tree.predict(X)
#             F_test[:, k] += lr * tree.predict(X_test)

#             trees[k].append(tree)

#         if (m + 1) % save_interval == 0:

#             # Compute accuracy
#             train_pred = np.argmax(softmax(F_train), axis=1)
#             test_pred = np.argmax(softmax(F_test), axis=1)

#             train_acc = np.mean(train_pred == y)
#             test_acc = np.mean(test_pred == y_test)

#             train_acc_history.append(train_acc)
#             test_acc_history.append(test_acc)

#             print(f"Iteration {m+1}: "
#                   f"Train Acc = {train_acc:.4f}, "
#                   f"Test Acc = {test_acc:.4f}")

#     return F0, trees, lr, train_acc_history, test_acc_history


# # ---------------------------------------
# # Train and Plot Function
# # ---------------------------------------
# def train_and_plot():

#     # Load Digits dataset
#     digits = load_digits()
#     X = digits.data
#     y = digits.target

#     # 80-20 split
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )

#     # Train model
#     F0, trees, lr, train_acc, test_acc = GradientBoostingClassification(
#         X_train, y_train,
#         X_test, y_test,
#         M=300,
#         max_depth=3,
#         min_samples_split=10,
#         min_samples_leaf=5,
#         lr=0.05,
#         save_interval=5
#     )

#     iterations = np.arange(5, 301, 5)

#     # Plot Accuracy vs M
#     plt.figure()
#     plt.plot(iterations, train_acc)
#     plt.plot(iterations, test_acc)

#     plt.xlabel("Number of Boosting Iterations (M)")
#     plt.ylabel("Accuracy")
#     plt.title("Train and Test Accuracy vs M")
#     plt.legend(["Train Accuracy", "Test Accuracy"])

#     plt.savefig("M_class.png", dpi=300)
#     plt.show()


# ---------------------------------------
# Main
# ---------------------------------------
if __name__ == "__main__":
    train_and_plot()

