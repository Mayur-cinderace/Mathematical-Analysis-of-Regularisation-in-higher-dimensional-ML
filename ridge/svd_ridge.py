import numpy as np

np.set_printoptions(precision=4, suppress=True)

# ============================================================
# 1. HIGH-DIMENSIONAL DATA GENERATION
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

print("High-dimensional data generated")
print(f"X shape: {X.shape}")
print("-" * 60)

# ============================================================
# 2. EIGEN-DECOMPOSITION OF XᵀX (CORE SVD STEP)
# ============================================================

XtX = X.T @ X

# Since XᵀX is symmetric positive semidefinite
eigvals, V = np.linalg.eigh(XtX)

# Sort eigenvalues descending
idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
V = V[:, idx]

# Keep only non-zero eigenvalues
tol = 1e-10
mask = eigvals > tol

eigvals = eigvals[mask]
V = V[:, mask]

# Singular values
S = np.sqrt(eigvals)

print("Number of non-zero singular values:", len(S))
print("Expected rank (≤ n):", n)
print("-" * 60)

# ============================================================
# 3. CONSTRUCT LEFT SINGULAR VECTORS U
# ============================================================

U = X @ V / S

# ============================================================
# 4. VERIFY SVD PROPERTIES
# ============================================================

print("UᵀU ≈ I:", np.allclose(U.T @ U, np.eye(len(S)), atol=1e-8))
print("VᵀV ≈ I:", np.allclose(V.T @ V, np.eye(len(S)), atol=1e-8))
print("-" * 60)

# ============================================================
# 5. RECONSTRUCTION CHECK
# ============================================================

X_recon = U @ np.diag(S) @ V.T
recon_error = np.linalg.norm(X - X_recon)

print("Reconstruction error ||X − UΣVᵀ||₂:", recon_error)
print("-" * 60)

# ============================================================
# 6. RIDGE REGRESSION VIA *CUSTOM* SVD
# ============================================================

def ridge_from_svd(U, S, V, y, lam):
    """
    Ridge regression using explicit SVD:

        β̂ = V diag( s_i / (s_i^2 + λ) ) Uᵀ y
    """
    shrinkage = S / (S**2 + lam)
    return V @ (shrinkage * (U.T @ y))

lam = 1.0
beta_ridge_svd = ridge_from_svd(U, S, V, y, lam)

print("Ridge solution via custom SVD computed")
print("||β̂||₂:", np.linalg.norm(beta_ridge_svd))
print("-" * 60)

# ============================================================
# 7. COMPARE WITH PRIMAL RIDGE SOLUTION
# ============================================================

beta_primal = np.linalg.solve(
    X.T @ X + lam * np.eye(d),
    X.T @ y
)

diff = np.linalg.norm(beta_primal - beta_ridge_svd)

print("||β_primal − β_svd||₂:", diff)
print("→ Solutions are numerically identical")
print("-" * 60)

# ============================================================
# 8. ESTIMATION ERROR
# ============================================================

err = np.linalg.norm(beta_ridge_svd - beta_true)

print("Estimation error ||β̂ − β*||₂:", err)
print("-" * 60)

# ============================================================
# 9. EFFECTIVE DIMENSION (SVD VIEW)
# ============================================================

def effective_dimension(S, lam):
    return np.sum(S**2 / (S**2 + lam))

for lam_test in [0.01, 0.1, 1.0, 10.0, 100.0]:
    d_eff = effective_dimension(S, lam_test)
    print(f"λ = {lam_test:<6} → effective dimension = {d_eff:.2f}")

print("-" * 60)
print("Custom SVD + ridge regression completed successfully.")
