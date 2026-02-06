import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# HIGH-DIMENSIONAL RIDGE REGRESSION
# A Complete Linear Algebra–Based Experimental Study
# ============================================================

np.set_printoptions(precision=4, suppress=True)

print("\n📐 High-Dimensional Ridge Regression")
print("A Complete Linear Algebra–Based Experimental Study")
print("Regime: d >> n\n")

# ============================================================
# 1. DATA GENERATION
# ============================================================

print("1) Dataset Generation")

np.random.seed(42)

n = 60
d = 500
s = 8
sigma = 0.1

X = np.random.randn(n, d)

beta_true = np.zeros(d)
beta_true[:s] = np.random.randn(s)

y = X @ beta_true + sigma * np.random.randn(n)

print(f"X shape = {X.shape}")
print(f"y shape = {y.shape}")
print(f"True sparsity (nonzeros in β*) = {s}\n")

# ============================================================
# 2. DATASET VISUALISATION (PCA)
# ============================================================

print("2) Dataset Visualisation via PCA")

X_centered = X - X.mean(axis=0)
U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

X_pca = U[:, :2] @ np.diag(S[:2])

plt.figure(figsize=(4, 3.6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7)
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.title("2D PCA Projection")
plt.tight_layout()
plt.show()

explained_var = (S**2) / np.sum(S**2)
cum_var = np.cumsum(explained_var)

plt.figure(figsize=(4, 3.6))
plt.plot(cum_var, marker="o")
plt.xlabel("Number of components")
plt.ylabel("Cumulative variance")
plt.title("Cumulative Explained Variance")
plt.tight_layout()
plt.show()

plt.figure(figsize=(4, 3.0))
plt.stem(beta_true)
plt.xlabel("Feature index")
plt.ylabel("Value")
plt.title("True parameter vector β* (sparse)")
plt.tight_layout()
plt.show()

# ============================================================
# 3. RANK DEFICIENCY
# ============================================================

print("3) Rank Deficiency and Ill-Posedness")

rank_X = np.linalg.matrix_rank(X)
null_dim = d - rank_X

print(f"rank(X) = {rank_X}")
print(f"ambient dimension d = {d}")
print(f"null space dimension = {null_dim}")
print("→ Unregularised least squares has infinitely many solutions\n")

# ============================================================
# 4. LOSS GEOMETRY
# ============================================================

print("4) Loss Geometry via XᵀX")

XtX = X.T @ X
eigvals = np.linalg.eigvalsh(XtX)

print(f"Smallest eigenvalue of XᵀX = {eigvals.min():.2e}")
print(f"Number of near-zero eigenvalues = {np.sum(eigvals < 1e-10)}\n")

# ============================================================
# 5. RIDGE REGRESSION FROM SCRATCH
# ============================================================

print("5) Ridge Regression from Scratch")

def ridge_regression(X, y, lam):
    return np.linalg.solve(
        X.T @ X + lam * np.eye(X.shape[1]),
        X.T @ y
    )

lam = 1.0
print(f"Using regularisation λ = {lam}")

beta_ridge = ridge_regression(X, y, lam)

# ============================================================
# 6. WELL-POSEDNESS
# ============================================================

print("\n6) Ridge Restores Well-Posedness")

min_eig_ridge = (eigvals + lam).min()
print(f"min eigenvalue of (XᵀX + λI) = {min_eig_ridge:.3f}")
print("→ Spectrum lifted away from zero\n")

# ============================================================
# 7. RIDGE SOLUTION & ERROR
# ============================================================

print("7) Ridge Solution and Estimation Error")

norm_beta = np.linalg.norm(beta_ridge)
err = np.linalg.norm(beta_ridge - beta_true)

print(f"||β̂_ridge||₂ = {norm_beta:.3f}")
print(f"||β̂_ridge − β*||₂ = {err:.3f}\n")

plt.figure(figsize=(4, 3.0))
plt.plot(beta_true, label="True β*", linewidth=2)
plt.plot(beta_ridge, label="Ridge estimate", alpha=0.7)
plt.legend()
plt.title("True vs Ridge Coefficients")
plt.tight_layout()
plt.show()

# ============================================================
# 8. EFFECTIVE DIMENSION
# ============================================================

print("8) Effective Dimension")

def effective_dimension(eigvals, lam):
    return np.sum(eigvals / (eigvals + lam))

lam_vals = [0.01, 0.1, 1, 10, 100]
d_eff = [effective_dimension(eigvals, l) for l in lam_vals]

df_eff = pd.DataFrame({
    "lambda": lam_vals,
    "effective_dimension": d_eff
})

print(df_eff, "\n")

# ============================================================
# 9. OLS COMPARISON
# ============================================================

print("9) Comparison with Ordinary Least Squares")

try:
    beta_ols = np.linalg.solve(X.T @ X, X.T @ y)
    ols_err = np.linalg.norm(beta_ols - beta_true)
    print(f"||β̂_OLS − β*||₂ = {ols_err:.2f}")
except np.linalg.LinAlgError:
    print("OLS failed: XᵀX is singular (expected when d >> n)")

# ============================================================
# 10. FINAL TAKEAWAY
# ============================================================

print("\n📌 Final Takeaway")

print("""
• High-dimensional least squares is ill-posed
• Rank deficiency creates flat loss geometry
• Ridge regularisation restores invertibility
• Conditioning improves via spectral lifting
• Effective dimension is controlled by λ
• All conclusions follow directly from linear algebra

This is ridge regression without black boxes.
""")
