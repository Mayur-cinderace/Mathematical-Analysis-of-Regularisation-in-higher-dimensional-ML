import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="High-Dimensional Ridge Regression",
    layout="wide"
)

st.title("📐 High-Dimensional Ridge Regression")
st.subheader("A Complete Linear Algebra–Based Experimental Study")

st.markdown("""
This interactive report demonstrates **ridge (L2) regularisation**
implemented **entirely from scratch** using linear algebra.

We explicitly work in the **high-dimensional regime**:
""")

st.latex(r"d \gg n")

st.markdown("""
The goal is to make **geometry, conditioning, and regularisation effects**
*visible and interpretable*.
""")

# ============================================================
# 1. DATA GENERATION
# ============================================================

st.header("1️⃣ Dataset Generation")

np.random.seed(42)

n = 60
d = 500
s = 8
sigma = 0.1

X = np.random.randn(n, d)
beta_true = np.zeros(d)
beta_true[:s] = np.random.randn(s)
y = X @ beta_true + sigma * np.random.randn(n)

st.code(
    f"X shape = {X.shape}\n"
    f"y shape = {y.shape}\n"
    f"True sparsity (nonzeros in β*) = {s}"
)

st.markdown("""
- Each row of **X** is a data point in \\(\\mathbb{R}^{500}\\)
- The true parameter **β\*** is sparse
- Noise is added to simulate a realistic regression setting
""")

# ============================================================
# 2. DATASET VISUALISATION
# ============================================================

st.header("2️⃣ Dataset Visualisation (High → Low Dimension)")

# PCA via SVD
X_centered = X - X.mean(axis=0)
U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

X_pca = U[:, :2] @ np.diag(S[:2])

fig, ax = plt.subplots(figsize=(4, 3.6))
ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7)
ax.set_xlabel("PC 1")
ax.set_ylabel("PC 2")
ax.set_title("2D PCA Projection")
fig.tight_layout()
st.pyplot(fig)

st.markdown("""
Even though the ambient dimension is 500, the data lives in a
**low-rank subspace (rank ≤ n)**.

PCA exposes the **same singular directions** that ridge later shrinks.
""")

# Variance explained
explained_var = (S**2) / np.sum(S**2)
cum_var = np.cumsum(explained_var)

fig, ax = plt.subplots(figsize=(4, 3.6))
ax.plot(cum_var, marker="o")
ax.set_xlabel("Number of components")
ax.set_ylabel("Cumulative variance")
ax.set_title("Cumulative Explained Variance")
fig.tight_layout()
st.pyplot(fig)

# True beta
fig, ax = plt.subplots(figsize=(4, 3.0))
ax.stem(beta_true)
ax.set_xlabel("Feature index")
ax.set_ylabel("Value")
ax.set_title("True parameter vector β* (sparse)")
fig.tight_layout()
st.pyplot(fig)

# ============================================================
# 3. RANK DEFICIENCY
# ============================================================

st.header("3️⃣ Rank Deficiency and Ill-Posedness")

rank_X = np.linalg.matrix_rank(X)
null_dim = d - rank_X

st.latex(r"\mathrm{rank}(X) \le \min(n, d)")

st.markdown(f"""
- **rank(X)** = {rank_X}  
- **ambient dimension (d)** = {d}  
- **null space dimension** = {null_dim}  

Unregularised least squares therefore has **infinitely many solutions**.
""")

# ============================================================
# 4. LOSS GEOMETRY
# ============================================================

st.header("4️⃣ Loss Geometry via $X^TX$")

XtX = X.T @ X
eigvals = np.linalg.eigvalsh(XtX)

st.code(
    f"Smallest eigenvalue of X^T X = {eigvals.min():.2e}\n"
    f"Number of near-zero eigenvalues = {np.sum(eigvals < 1e-10)}"
)

st.markdown("""
Near-zero eigenvalues correspond to **flat directions**
in the loss surface, making the problem ill-conditioned.
""")

# ============================================================
# 5. RIDGE REGRESSION
# ============================================================

st.header("5️⃣ Ridge Regression from Scratch")

st.latex(r"\min_\beta \; \|y - X\beta\|_2^2 + \lambda \|\beta\|_2^2")
st.latex(r"(X^TX + \lambda I)\beta = X^Ty")

def ridge_regression(X, y, lam):
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)

lam = st.slider("Regularisation strength λ", 0.01, 10.0, 1.0)
beta_ridge = ridge_regression(X, y, lam)

# ============================================================
# 6. WELL-POSEDNESS
# ============================================================

st.header("6️⃣ Ridge Restores Well-Posedness")

st.latex(
    rf"\min \lambda_i(X^TX + \lambda I) = {(eigvals + lam).min():.3f}"
)

st.markdown("""
Adding \\(\\lambda I\\) **lifts the entire spectrum**,
ensuring invertibility and numerical stability.
""")

# ============================================================
# 7. RIDGE SOLUTION & ERROR
# ============================================================

st.header("7️⃣ Ridge Solution and Error")

st.latex(rf"\|\hat{{\beta}}_{{ridge}}\|_2 = {np.linalg.norm(beta_ridge):.3f}")

err = np.linalg.norm(beta_ridge - beta_true)
st.latex(rf"\|\hat{{\beta}}_{{ridge}} - \beta^*\|_2 = {err:.3f}")

fig, ax = plt.subplots(figsize=(4, 3.0))
ax.plot(beta_true, label="True β*", linewidth=2)
ax.plot(beta_ridge, label="Ridge estimate", alpha=0.7)
ax.legend()
ax.set_title("True vs Ridge Coefficients")
fig.tight_layout()
st.pyplot(fig)

# ============================================================
# 8. EFFECTIVE DIMENSION
# ============================================================

st.header("8️⃣ Effective Dimension")

def effective_dimension(eigvals, lam):
    return np.sum(eigvals / (eigvals + lam))

lam_vals = [0.01, 0.1, 1, 10, 100]
d_eff = [effective_dimension(eigvals, l) for l in lam_vals]

st.dataframe(pd.DataFrame({
    "λ": lam_vals,
    "Effective dimension": d_eff
}))

# ============================================================
# 9. OLS COMPARISON
# ============================================================

st.header("9️⃣ Comparison with OLS")

try:
    beta_ols = np.linalg.solve(X.T @ X, X.T @ y)
    ols_err = np.linalg.norm(beta_ols - beta_true)
    st.latex(rf"\|\hat{{\beta}}_{{OLS}} - \beta^*\|_2 = {ols_err:.2f}")
except np.linalg.LinAlgError:
    st.error("OLS failed: XᵀX is singular (as expected when d ≫ n)")

# ============================================================
# 10. FINAL TAKEAWAY
# ============================================================

st.header("📌 Final Takeaway")

st.markdown("""
- High-dimensional least squares is **ill-posed**
- Rank deficiency creates **flat loss geometry**
- Ridge regularisation restores **invertibility**
- Conditioning improves via **spectral lifting**
- Effective dimension is controlled by λ
- All conclusions follow directly from **linear algebra**

This is ridge regression **without black boxes**.
""")
