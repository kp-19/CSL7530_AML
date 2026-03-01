import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from GradientBoostingRegression import GradientBoostingRegression, inference_regression
from GradientBoostingClassification import GradientBoostingClassifier, inference_classification
from sklearn.datasets import fetch_california_housing, load_iris, load_digits


M_vals_regression = [50, 100, 200]
M_vals_classification = [10, 25, 50]
lr_vals = [0.01, 0.05, 0.1, 1.0]
tree_depth_vals = [1, 3, 10]
min_samples_leaf_vals = [1, 5, 10]

def load_california_housing():
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

    return X_train, X_test, y_train, y_test

def load_iris_dataset():
    data = load_iris()

    X = data.data          # shape (150, 4)
    y = data.target        # shape (150,)

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Classes:", np.unique(y))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y  # keeps class balance
    )

    return X_train, X_test, y_train, y_test


def load_digits_dataset():
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

    return X_train, X_test, y_train, y_test

def annotate_line_plot(x_vals, y_vals, fmt="{:.3f}", offset=5):

    for x, y in zip(x_vals, y_vals):
        plt.annotate(fmt.format(y),
                     (x, y),
                     textcoords="offset points",
                     xytext=(0, offset),
                     ha='center')
        
def annotate_bars(bars, fmt="{:.2f}", offset=3):

    for bar in bars:
        height = bar.get_height()
        plt.annotate(fmt.format(height),
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, offset),
                     textcoords="offset points",
                     ha='center',
                     va='bottom')

def regression_M(M_vals_regression):

    X_train, X_test, y_train, y_test = load_california_housing()

    train_mse_list = []
    test_mse_list = []
    largest_loss_history = None
    largest_M = max(M_vals_regression)

    for m in M_vals_regression:

        f0, trees, lr, loss_history = GradientBoostingRegression(
            X_train, y_train,
            M=m,
            max_depth=3,
            min_samples_split=10,
            min_samples_leaf=5,
            lr=0.05,
            save_loss_interval=1
        )

        y_pred_train = inference_regression(X_train, f0, trees, lr)
        y_pred_test = inference_regression(X_test, f0, trees, lr)

        train_mse = np.mean((y_train - y_pred_train) ** 2)
        test_mse = np.mean((y_test - y_pred_test) ** 2)

        train_mse_list.append(train_mse)
        test_mse_list.append(test_mse)

        if m == largest_M:
            largest_loss_history = loss_history

    # Plot training loss history (largest M)
    plt.figure()
    plt.plot(range(1, largest_M + 1), largest_loss_history)
    plt.xlabel("Iterations")
    plt.ylabel("Training MSE")
    plt.title(f"Training Loss vs Iterations (M={largest_M})")
    plt.show()

    # Plot final MSE vs M
    plt.figure()
    plt.plot(M_vals_regression, train_mse_list, marker='o', label="Train MSE")
    plt.plot(M_vals_regression, test_mse_list, marker='o', label="Test MSE")

    annotate_line_plot(M_vals_regression, train_mse_list)
    annotate_line_plot(M_vals_regression, test_mse_list)

    plt.xlabel("M")
    plt.ylabel("MSE")
    plt.title("Train/Test MSE vs M")
    plt.legend()
    plt.show()

def regression_lr(lr_vals):

    X_train, X_test, y_train, y_test = load_california_housing()

    train_mse_list = []
    test_mse_list = []

    for lr in lr_vals:

        f0, trees, _, _ = GradientBoostingRegression(
            X_train, y_train,
            M=100,
            max_depth=3,
            min_samples_split=10,
            min_samples_leaf=5,
            lr=lr
        )

        y_pred_train = inference_regression(X_train, f0, trees, lr)
        y_pred_test = inference_regression(X_test, f0, trees, lr)

        train_mse_list.append(np.mean((y_train - y_pred_train) ** 2))
        test_mse_list.append(np.mean((y_test - y_pred_test) ** 2))

    plt.figure()
    plt.plot(lr_vals, train_mse_list, marker='o', label="Train MSE")
    plt.plot(lr_vals, test_mse_list, marker='o', label="Test MSE")

    annotate_line_plot(lr_vals, train_mse_list)
    annotate_line_plot(lr_vals, test_mse_list)

    plt.xlabel("Learning Rate")
    plt.ylabel("MSE")
    plt.title("Train/Test MSE vs Learning Rate")
    plt.legend()
    plt.show()

def regression_tree_depth(tree_depth_vals):

    X_train, X_test, y_train, y_test = load_california_housing()

    train_mse_list = []
    test_mse_list = []

    for depth in tree_depth_vals:

        f0, trees, lr, _ = GradientBoostingRegression(
            X_train, y_train,
            M=100,
            max_depth=depth,
            min_samples_split=10,
            min_samples_leaf=5,
            lr=0.05
        )

        y_pred_train = inference_regression(X_train, f0, trees, lr)
        y_pred_test = inference_regression(X_test, f0, trees, lr)

        train_mse_list.append(np.mean((y_train - y_pred_train) ** 2))
        test_mse_list.append(np.mean((y_test - y_pred_test) ** 2))

    plt.figure()
    plt.plot(tree_depth_vals, train_mse_list, marker='o', label="Train MSE")
    plt.plot(tree_depth_vals, test_mse_list, marker='o', label="Test MSE")
    
    annotate_line_plot(tree_depth_vals, train_mse_list)
    annotate_line_plot(tree_depth_vals, test_mse_list)

    plt.xlabel("Tree Depth")
    plt.ylabel("MSE")
    plt.title("Train/Test MSE vs Tree Depth")
    plt.legend()
    plt.show()

def regression_tree_min_samples_leaf(min_samples_leaf_vals):

    X_train, X_test, y_train, y_test = load_california_housing()

    train_mse_list = []
    test_mse_list = []

    for min_leaf in min_samples_leaf_vals:

        f0, trees, lr, _ = GradientBoostingRegression(
            X_train, y_train,
            M=100,
            max_depth=3,
            min_samples_split=10,
            min_samples_leaf=min_leaf,
            lr=0.05
        )

        y_pred_train = inference_regression(X_train, f0, trees, lr)
        y_pred_test = inference_regression(X_test, f0, trees, lr)

        train_mse_list.append(np.mean((y_train - y_pred_train) ** 2))
        test_mse_list.append(np.mean((y_test - y_pred_test) ** 2))

    plt.figure()
    plt.plot(min_samples_leaf_vals, train_mse_list, marker='o', label="Train MSE")
    plt.plot(min_samples_leaf_vals, test_mse_list, marker='o', label="Test MSE")

    annotate_line_plot(min_samples_leaf_vals, train_mse_list)
    annotate_line_plot(min_samples_leaf_vals, test_mse_list)

    plt.xlabel("Min Samples Leaf")
    plt.ylabel("MSE")
    plt.title("Train/Test MSE vs Min Samples Leaf")
    plt.legend()
    plt.show()
    
def classification_M(M_vals_classification):

    X_train, X_test, y_train, y_test = load_digits_dataset()

    train_acc_list = []
    test_acc_list = []
    largest_M = max(M_vals_classification)
    largest_loss_history = None

    for m in M_vals_classification:

        trees, loss_history = GradientBoostingClassifier(
            X_train, y_train,
            M=m,
            max_depth=3,
            min_samples_leaf=5,
            lr=0.1,
            save_loss_interval=1
        )

        y_pred_train = inference_classification(X_train, trees, 0.1)
        y_pred_test = inference_classification(X_test, trees, 0.1)

        train_acc_list.append(np.mean(y_train == y_pred_train))
        test_acc_list.append(np.mean(y_test == y_pred_test))

        if m == largest_M:
            largest_loss_history = loss_history

    # Plot loss history
    plt.figure()
    plt.plot(range(1, largest_M + 1), largest_loss_history)
    plt.xlabel("Iterations")
    plt.ylabel("Training Loss")
    plt.title(f"Training Loss vs Iterations (M={largest_M})")
    plt.show()

    # Bar plot accuracy
    x = np.arange(len(M_vals_classification))
    width = 0.35

    plt.figure()
    bars1 = plt.bar(x - width/2, train_acc_list, width, label="Train")
    bars2 = plt.bar(x + width/2, test_acc_list, width, label="Test")

    annotate_bars(bars1)
    annotate_bars(bars2)

    plt.xticks(x, M_vals_classification)
    plt.xlabel("M")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs M")
    plt.legend()
    plt.show()

def classification_lr(lr_vals):

    X_train, X_test, y_train, y_test = load_digits_dataset()

    train_acc_list = []
    test_acc_list = []

    for lr in lr_vals:

        trees, _ = GradientBoostingClassifier(
            X_train, y_train,
            M=20,
            max_depth=3,
            min_samples_leaf=5,
            lr=lr
        )

        y_pred_train = inference_classification(X_train, trees, lr)
        y_pred_test = inference_classification(X_test, trees, lr)

        train_acc_list.append(np.mean(y_train == y_pred_train))
        test_acc_list.append(np.mean(y_test == y_pred_test))

    x = np.arange(len(lr_vals))
    width = 0.35

    plt.figure()
    bars1 = plt.bar(x - width/2, train_acc_list, width, label="Train")
    bars2 = plt.bar(x + width/2, test_acc_list, width, label="Test")

    annotate_bars(bars1)
    annotate_bars(bars2)

    plt.xticks(x, lr_vals)
    plt.xlabel("Learning Rate")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Learning Rate")
    plt.legend()
    plt.show()


def classification_tree_depth(tree_depth_vals):

    X_train, X_test, y_train, y_test = load_digits_dataset()

    train_acc_list = []
    test_acc_list = []

    for depth in tree_depth_vals:

        trees, _ = GradientBoostingClassifier(
            X_train, y_train,
            M=20,
            max_depth=depth,
            min_samples_split=2,
            min_samples_leaf=5,
            lr=0.1
        )

        y_pred_train = inference_classification(X_train, trees, 0.1)
        y_pred_test = inference_classification(X_test, trees, 0.1)

        train_acc_list.append(np.mean(y_train == y_pred_train))
        test_acc_list.append(np.mean(y_test == y_pred_test))

    x = np.arange(len(tree_depth_vals))
    width = 0.35

    plt.figure()
    bars1 = plt.bar(x - width/2, train_acc_list, width, label="Train")
    bars2 = plt.bar(x + width/2, test_acc_list, width, label="Test")

    annotate_bars(bars1)
    annotate_bars(bars2)

    plt.xticks(x, tree_depth_vals)
    plt.xlabel("Tree Depth")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Tree Depth")
    plt.legend()
    plt.show()

def classification_tree_min_samples_leaf(min_samples_leaf_vals):

    X_train, X_test, y_train, y_test = load_digits_dataset()

    train_acc_list = []
    test_acc_list = []

    for min_leaf in min_samples_leaf_vals:

        trees, _ = GradientBoostingClassifier(
            X_train, y_train,
            M=50,
            max_depth=3,
            min_samples_split=2,
            min_samples_leaf=min_leaf,
            lr=0.1
        )

        y_pred_train = inference_classification(X_train, trees, 0.1)
        y_pred_test = inference_classification(X_test, trees, 0.1)

        train_acc_list.append(np.mean(y_train == y_pred_train))
        test_acc_list.append(np.mean(y_test == y_pred_test))

    x = np.arange(len(min_samples_leaf_vals))
    width = 0.35

    plt.figure()
    bars1 = plt.bar(x - width/2, train_acc_list, width, label="Train")
    bars2 = plt.bar(x + width/2, test_acc_list, width, label="Test")

    annotate_bars(bars1)
    annotate_bars(bars2)

    plt.xticks(x, min_samples_leaf_vals)
    plt.xlabel("Min Samples Leaf")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Min Samples Leaf")
    plt.legend()
    plt.show()


if __name__ == "__main__":

    # regression_M(M_vals_regression)
    # regression_lr(lr_vals)
    # regression_tree_depth(tree_depth_vals)
    # regression_tree_min_samples_leaf(min_samples_leaf_vals)
    # classification_M(M_vals_classification)
    # classification_lr(lr_vals)
    # classification_tree_depth(tree_depth_vals)
    # classification_tree_min_samples_leaf(min_samples_leaf_vals)

