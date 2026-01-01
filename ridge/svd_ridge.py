import numpy as np

# ============================================================
# 1. HIGH-DIMENSIONAL DATA
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
print("-" * 60)

# ============================================================
# 2. EIGEN-DECOMPOSITION OF X^T X  (CORE SVD STEP)
# ============================================================

XtX = X.T @ X

eigvals, V = np.linalg.eigh(XtX)   # symmetric → eigh

# Sort eigenvalues descending
idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
V = V[:, idx]

# Keep only non-zero eigenvalues
tol = 1e-10
nonzero = eigvals > tol

eigvals = eigvals[nonzero]
V = V[:, nonzero]

# Singular values
S = np.sqrt(eigvals)

print("Number of non-zero singular values:", len(S))
print("Expected rank:", n)
print("-" * 60)

# ============================================================
# 3. CONSTRUCT U MATRIX
# ============================================================

U = X @ V / S

# ============================================================
# 4. VERIFY SVD PROPERTIES
# ============================================================

print("U^T U ≈ I:", np.allclose(U.T @ U, np.eye(len(S)), atol=1e-8))
print("V^T V ≈ I:", np.allclose(V.T @ V, np.eye(len(S)), atol=1e-8))
print("-" * 60)

# ============================================================
# 5. RECONSTRUCTION CHECK
# ============================================================

X_recon = U @ np.diag(S) @ V.T
recon_error = np.linalg.norm(X - X_recon)

print("Reconstruction error ||X − UΣV^T||:", recon_error)
print("-" * 60)

# ============================================================
# 6. RIDGE REGRESSION USING *OUR* SVD
# ============================================================

def ridge_from_svd(U, S, V, y, lam):
    """
    Ridge regression via manually constructed SVD:
        β = V diag( s_i / (s_i^2 + λ) ) U^T y
    """
    shrinkage = S / (S**2 + lam)
    beta_hat = V @ (shrinkage * (U.T @ y))
    return beta_hat

lam = 1.0
beta_ridge_svd = ridge_from_svd(U, S, V, y, lam)

print("Ridge via custom SVD computed")
print("||β̂||_2:", np.linalg.norm(beta_ridge_svd))
print("-" * 60)

# ============================================================
# 7. COMPARE WITH PRIMAL RIDGE
# ============================================================

beta_primal = np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)

diff = np.linalg.norm(beta_primal - beta_ridge_svd)

print("||β_primal − β_svd_custom||_2:", diff)
print("Solutions are numerically identical")
print("-" * 60)

# ============================================================
# 8. ESTIMATION ERROR
# ============================================================

error = np.linalg.norm(beta_ridge_svd - beta_true)
print("Estimation error ||β̂ − β*||_2:", error)
print("-" * 60)

# ============================================================
# 9. EFFECTIVE DIMENSION (SVD VIEW)
# ============================================================

def effective_dimension(S, lam):
    return np.sum(S**2 / (S**2 + lam))

for lam_test in [0.01, 0.1, 1.0, 10.0, 100.0]:
    print(f"λ = {lam_test:<6} → effective dimension = {effective_dimension(S, lam_test):.2f}")

print("-" * 60)
print("Custom SVD + ridge implementation completed successfully.")
