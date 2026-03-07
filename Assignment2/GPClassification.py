import numpy as np

def rbf_kernel(X1, X2, l = 1.0, sigma_f = 1.0):

    dis_sq = (np.sum(X1**2, axis=1).reshape(-1,1) + np.sum(X2**2, axis=1) -2*np.dot(X1, X2.T))

    out = sigma_f**2 * np.exp(-(0.5/l**2)*dis_sq)

    return out

def compute_fhat(X, y, tolerance = 1e-6, max_iters = 50):

    K = rbf_kernel(X, X)
    n = X.shape[0]

    f = np.zeros(n)

    for i in range(max_iters):
        f_old = f.copy() 

        pi = 1 / (1 + np.exp(-y * f))
        grad = y * (1 - pi)

        W = pi*(1-pi)

        W_sqrt = np.sqrt(W)
        B = np.eye(n) + W_sqrt[:,None]*K*W_sqrt[None,:]

        L = np.linalg.cholesky(B)

        b = W*f + grad

        W_sqrt_Kb = W_sqrt * (K@b)

        tmp = np.linalg.solve(L, W_sqrt_Kb)
        tmp = np.linalg.solve(L.T, tmp)

        a = b - W_sqrt*tmp

        f = K@a
        if np.linalg.norm(f - f_old) < tolerance : 
            break

    ll = np.sum(-np.log(1 + np.exp(-y*f)))

    lm = -0.5*a.T@f + ll - np.sum(np.log(np.diag(L)))

    return f, a, ll, lm

def GPClassification(X, y, X_test, f_hat):
    
    K = rbf_kernel(X, X)
    n = X.shape[0]

    pi = 1 / (1 + np.exp(-y * f_hat))
    W = pi * (1-pi)

    W_sqrt = np.sqrt(W)
    B = np.eye(n) + W_sqrt[:,None]*K*W_sqrt[None,:]
    L = np.linalg.cholesky(B)

    grad = y * (1 - pi)

    k_star = rbf_kernel(X, X_test)

    # predictive mean:
    f_bar = k_star.T@grad

    v = np.linalg.solve(L, W_sqrt[:, None]*k_star)

    k_xx = rbf_kernel(X_test, X_test)
    var = np.diag(k_xx - v.T@v)

    # Approximation for the integration step:
    kappa = 1/np.sqrt(1+np.pi*var / 8)
    pi_star = 1/(1+np.exp(-kappa*f_bar))

    return pi_star, f_bar, var





