import numpy as np

# ============================================================
# 1. DATA GENERATION (SAME SETUP AS RIDGE)
# ============================================================

np.random.seed(42)

n = 60
d = 500
s = 8
sigma = 0.1

X = np.random.randn(n, d)

beta_true = np.zeros(d)
beta_true[:s] = np.random.randn(s)

y = X @ beta_true + sigma * np.random.randn(n)

print("Data generated")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("-" * 50)

# ============================================================
# 2. SOFT-THRESHOLDING OPERATOR
# ============================================================

def soft_threshold(z, lam):
    """
    Soft-thresholding operator:
        S(z, λ)
    """
    if z > lam:
        return z - lam
    elif z < -lam:
        return z + lam
    else:
        return 0.0

# ============================================================
# 3. LASSO VIA COORDINATE DESCENT (FROM SCRATCH)
# ============================================================

def lasso_coordinate_descent(X, y, lam, max_iter=1000, tol=1e-6):
    """
    Solve LASSO using coordinate descent.

    Objective:
        (1/2)||y - Xβ||^2 + λ||β||_1
    """
    n, d = X.shape
    beta = np.zeros(d)

    # Precompute column norms
    X_col_norms = np.sum(X ** 2, axis=0)

    for it in range(max_iter):
        beta_old = beta.copy()

        for j in range(d):
            # Partial residual (excluding j-th feature)
            r_j = y - X @ beta + X[:, j] * beta[j]

            # Coordinate-wise update
            rho = X[:, j].T @ r_j
            beta[j] = soft_threshold(rho, lam) / X_col_norms[j]

        # Convergence check
        if np.linalg.norm(beta - beta_old) < tol:
            break

    return beta

# ============================================================
# 4. RUN LASSO
# ============================================================

lam = 0.05

beta_lasso = lasso_coordinate_descent(X, y, lam)

print("LASSO solved via coordinate descent")
print("||β̂_lasso||_2:", np.linalg.norm(beta_lasso))
print("-" * 50)

# ============================================================
# 5. SPARSITY ANALYSIS
# ============================================================

nonzeros = np.sum(np.abs(beta_lasso) > 1e-6)

print("True non-zero coefficients:", s)
print("LASSO non-zero coefficients:", nonzeros)
print("-" * 50)

# ============================================================
# 6. ESTIMATION ERROR
# ============================================================

error = np.linalg.norm(beta_lasso - beta_true)

print("LASSO estimation error ||β̂ − β*||_2:", error)
print("-" * 50)

print("L1 regularisation (LASSO) from scratch completed successfully.")
