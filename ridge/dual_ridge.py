import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# HIGH-DIMENSIONAL RIDGE REGULARISATION
# Complete Linear Algebra Analysis (Primal & Dual)
# ============================================================

np.set_printoptions(precision=4, suppress=True)

print("\n📐 High-Dimensional Ridge Regularisation")
print("Complete Linear Algebra Analysis (Primal & Dual)")
print("Regime: d >> n\n")

# ============================================================
# 1. DATA GENERATION
# ============================================================

print("1) Data Generation")

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
print(f"y shape = {y.shape}\n")

# ============================================================
# 2. RANK & NULL SPACE
# ============================================================

print("2) Rank Deficiency and Null Space")

rank_X = np.linalg.matrix_rank(X)
null_dim = d - rank_X

print(f"rank(X) = {rank_X}")
print(f"ambient dimension d = {d}")
print(f"null space dimension = {null_dim}\n")

# ============================================================
# 3. SPECTRAL ANALYSIS OF XᵀX
# ============================================================

print("3) Spectrum of XᵀX")

XtX = X.T @ X
eigvals = np.linalg.eigvalsh(XtX)

min_eig = eigvals.min()
num_zero = np.sum(eigvals < 1e-10)

print(f"Smallest eigenvalue of XᵀX = {min_eig:.2e}")
print(f"Number of (near) zero eigenvalues = {num_zero}\n")

# ============================================================
# 4. RIDGE DEFINITIONS
# ============================================================

print("4) Ridge Regression Formulations")

print("Primal ridge:")
print("(XᵀX + λI) β = Xᵀ y")

print("\nDual ridge:")
print("β = Xᵀ (X Xᵀ + λI)⁻¹ y\n")

# ============================================================
# 5. PRIMAL & DUAL IMPLEMENTATIONS
# ============================================================

def ridge_primal(X, y, lam):
    return np.linalg.solve(
        X.T @ X + lam * np.eye(X.shape[1]),
        X.T @ y
    )

def ridge_dual(X, y, lam):
    alpha = np.linalg.solve(
        X @ X.T + lam * np.eye(X.shape[0]),
        y
    )
    return X.T @ alpha

lam = 1.0
print(f"Using regularisation λ = {lam}\n")

beta_primal = ridge_primal(X, y, lam)
beta_dual = ridge_dual(X, y, lam)

# ============================================================
# 6. PRIMAL ≡ DUAL (NUMERICAL)
# ============================================================

print("5) Primal ≡ Dual: Numerical Equivalence")

diff = np.linalg.norm(beta_primal - beta_dual)

print(f"||β_primal − β_dual||₂ = {diff:.2e}")
print("→ Agreement up to floating-point precision\n")

# ============================================================
# 7. CONDITION NUMBER ANALYSIS
# ============================================================

print("6) Conditioning and Numerical Stability")

nonzero_eigs = eigvals[eigvals > 1e-10]
cond_unreg = nonzero_eigs.max() / nonzero_eigs.min()

print(f"Condition number of XᵀX (restricted) = {cond_unreg:.4e}\n")

def cond_ridge(eigvals, lam):
    e = eigvals + lam
    return e.max() / e.min()

lams = np.array([1e-3, 1e-2, 1e-1, 1, 10, 100])
conds = [cond_ridge(eigvals, l) for l in lams]

df_cond = pd.DataFrame({
    "lambda": lams,
    "cond(XᵀX + λI)": conds
})

print("Condition number vs λ:")
print(df_cond, "\n")

plt.figure()
plt.plot(lams, conds, marker="o")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("λ")
plt.ylabel("Condition number")
plt.title("Effect of Ridge Regularisation on Conditioning")
plt.tight_layout()
plt.show()

# ============================================================
# 8. ESTIMATION ERROR
# ============================================================

print("7) Estimation Error")

err = np.linalg.norm(beta_primal - beta_true)

print(f"||β̂_ridge − β*||₂ = {err:.4f}")
print("→ Bias introduced by ridge regularisation\n")

# ============================================================
# 9. COMPUTATIONAL COST
# ============================================================

print("8) Computational Perspective")

print(f"Primal inversion: {d} × {d}")
print(f"Dual inversion:   {n} × {n}")

print("\nSince d >> n, the dual formulation is computationally preferable.\n")

# ============================================================
# 10. FINAL TAKEAWAY
# ============================================================

print("📌 Final Takeaway")

print("""
• High-dimensional least squares is ill-posed
• Ridge regularisation restores invertibility
• Regularisation lifts the spectrum away from zero
• Conditioning improves monotonically with λ
• Primal and dual ridge are exactly equivalent
• All conclusions follow directly from linear algebra
""")
