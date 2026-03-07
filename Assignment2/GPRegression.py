import numpy as np

def rbf_kernel(X1, X2, l = 1.0, sigma_f = 1.0):

    dis_sq = (np.sum(X1**2, axis=1).reshape(-1,1) + np.sum(X2**2, axis=1) -2*np.dot(X1, X2.T))

    out = sigma_f**2 * np.exp(-(0.5/l**2)*dis_sq)

    return out

def GPRegression(X, y, X_test, l=1.0, sigma_f=1.0, sigma_noise=0.1):

    K = rbf_kernel(X, X, l=l, sigma_f=sigma_f)
    n = X.shape[0]
    y = y.reshape(-1,1)

    Ky = K + (sigma_noise**2) * np.eye(n)   # Add noise term

    L = np.linalg.cholesky(Ky)      # Cholesky decomposition

    z = np.linalg.solve(L, y)
    alpha = np.linalg.solve(L.T, z)

    k_star = rbf_kernel(X, X_test, l=l, sigma_f=sigma_f)

    f_star = k_star.T @ alpha

    v = np.linalg.solve(L, k_star)

    k_xx = rbf_kernel(X_test, X_test, l=l, sigma_f=sigma_f)

    cov = k_xx - v.T@v

    log_likelihood = (-0.5*y.T@alpha) - np.sum(np.log(np.diag(L))) - (n/2)*np.log(2*np.pi)

    return f_star, cov, log_likelihood

# if __name__ == "__main__":

#     X = np.array([
#     [-2,-2],
#     [-1,1],
#     [1,-1],
#     [2,2]
#     ])

#     y = np.array([1, -1, 2, 0]).reshape(-1,1)

#     x_star = np.array([0,0])

#     mean, var, ll = GPRegression(X, y, x_star)

#     print("Mean function: ", mean)
#     print("Variance: ", var)
#     print("log likelihood: ", ll)
