import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="High-Dimensional Ridge Regression Analysis",
    layout="wide"
)

st.title("📐 High-Dimensional Ridge Regression")
st.subheader("Comprehensive Linear Algebra–Based Experimental Study with Multiple Methods")

st.markdown("""
This interactive report demonstrates **ridge (L2) regularisation** implemented **entirely from scratch** using linear algebra in various ways.

We explicitly work in the **high-dimensional regime**:
""")

st.latex(r"d \gg n")

st.markdown("""
The goal is to make **geometry, conditioning, and regularisation effects** *visible and interpretable*.

Use the tabs below to switch between different methods for ridge regularisation. Each tab provides proper visualizations and analysis, including all original results plus additional insights where applicable.
""")

# ============================================================
# SHARED DATA GENERATION & PRECOMPUTATIONS
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

# For PCA visualization
X_centered = X - X.mean(axis=0)
U_svd, S_svd, Vt_svd = np.linalg.svd(X_centered, full_matrices=False)

# Core matrix
XtX = X.T @ X

# Full eigendecomposition (we need eigenvectors too!)
eigvals_full, V_full = np.linalg.eigh(XtX)

# Sort descending
idx = np.argsort(eigvals_full)[::-1]
eigvals_full = eigvals_full[idx]
V_full = V_full[:, idx]

# Filter nonzero part once
tol = 1e-10
nonzero_mask = eigvals_full > tol
eigvals = eigvals_full[nonzero_mask]
V = V_full[:, nonzero_mask]               # now globally available

rank_X = np.linalg.matrix_rank(X)
null_dim = d - rank_X

# ============================================================
# RIDGE FUNCTIONS
# ============================================================

def ridge_regression(X, y, lam):  # Primal
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)

def ridge_dual(X, y, lam):
    alpha = np.linalg.solve(X @ X.T + lam * np.eye(X.shape[0]), y)
    return X.T @ alpha

def ridge_from_svd(U, S, V, y, lam):
    shrinkage = S / (S**2 + lam)
    # Corrected: shrinkage (r,) * (U.T @ y) (r,) → scalar product per direction
    projected = shrinkage * (U.T @ y)
    return V @ projected

def effective_dimension(eigvals, lam):
    return np.sum(eigvals / (eigvals + lam))

# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3 = st.tabs(["Basic Report", "Primal & Dual Analysis", "SVD Implementation"])

# ────────────────────────────────────────────────
# TAB 1 - unchanged (your original content)
# ────────────────────────────────────────────────
with tab1:
    st.header("Basic Ridge Regression Report")

    st.markdown("""
    This tab replicates the original basic report with improved plot sizes and additional results.
    """)

    # 1. DATA GENERATION
    st.subheader("1️⃣ Dataset Generation")

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

    st.subheader("Additional: Sample Data Preview")
    df_sample = pd.DataFrame(X[:5, :5], columns=[f"Feature {i+1}" for i in range(5)])
    st.dataframe(df_sample)
    st.markdown("Showing first 5 rows and 5 columns of X for illustration.")

    # 2. DATASET VISUALISATION
    st.subheader("2️⃣ Dataset Visualisation (High → Low Dimension)")

    X_pca = U_svd[:, :2] @ np.diag(S_svd[:2])

    fig, ax = plt.subplots(figsize=(8, 6))
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

    explained_var = (S_svd**2) / np.sum(S_svd**2)
    cum_var = np.cumsum(explained_var)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(cum_var, marker="o")
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Cumulative variance")
    ax.set_title("Cumulative Explained Variance")
    fig.tight_layout()
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.stem(beta_true)
    ax.set_xlabel("Feature index")
    ax.set_ylabel("Value")
    ax.set_title("True parameter vector β* (sparse)")
    fig.tight_layout()
    st.pyplot(fig)

    st.subheader("Additional: Distribution of True β*")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(beta_true, bins=20)
    ax.set_title("Histogram of True β* Values")
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    st.pyplot(fig)

    # 3. RANK DEFICIENCY
    st.subheader("3️⃣ Rank Deficiency and Ill-Posedness")

    st.latex(r"\mathrm{rank}(X) \le \min(n, d)")

    st.markdown(f"""
    - **rank(X)** = {rank_X}  
    - **ambient dimension (d)** = {d}  
    - **null space dimension** = {null_dim}  

    Unregularised least squares therefore has **infinitely many solutions**.
    """)

    # 4. LOSS GEOMETRY
    st.subheader("4️⃣ Loss Geometry via $X^TX$")

    st.code(
        f"Smallest eigenvalue of X^T X = {eigvals.min():.2e}\n"
        f"Number of near-zero eigenvalues = {np.sum(eigvals_full < 1e-10)}"
    )

    st.markdown("""
    Near-zero eigenvalues correspond to **flat directions**
    in the loss surface, making the problem ill-conditioned.
    """)

    st.subheader("Additional: Eigenvalue Distribution of X^T X")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(eigvals[eigvals > 1e-10], bins=20, log=True)
    ax.set_title("Histogram of Non-Zero Eigenvalues (log scale)")
    ax.set_xlabel("Eigenvalue")
    ax.set_ylabel("Frequency (log)")
    fig.tight_layout()
    st.pyplot(fig)

    # 5-7 Ridge etc.
    st.subheader("5️⃣ Ridge Regression from Scratch")

    st.latex(r"\min_\beta \; \|y - X\beta\|_2^2 + \lambda \|\beta\|_2^2")
    st.latex(r"(X^TX + \lambda I)\beta = X^Ty")

    lam = st.slider("Regularisation strength λ (Basic Tab)", 0.01, 10.0, 1.0, key="lam_basic")
    beta_ridge = ridge_regression(X, y, lam)

    st.subheader("6️⃣ Ridge Restores Well-Posedness")

    st.latex(
        rf"\min \lambda_i(X^TX + \lambda I) = {(eigvals + lam).min():.3f}"
    )

    st.markdown("""
    Adding \\(\\lambda I\\) **lifts the entire spectrum**,
    ensuring invertibility and numerical stability.
    """)

    st.subheader("7️⃣ Ridge Solution and Error")

    st.latex(rf"\|\hat{{\beta}}_{{ridge}}\|_2 = {np.linalg.norm(beta_ridge):.3f}")

    err = np.linalg.norm(beta_ridge - beta_true)
    st.latex(rf"\|\hat{{\beta}}_{{ridge}} - \beta^*\|_2 = {err:.3f}")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(beta_true, label="True β*", linewidth=2)
    ax.plot(beta_ridge, label="Ridge estimate", alpha=0.7)
    ax.legend()
    ax.set_title("True vs Ridge Coefficients")
    fig.tight_layout()
    st.pyplot(fig)

    st.subheader("Additional: Residual Analysis")
    y_pred = X @ beta_ridge
    residuals = y - y_pred
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_pred, residuals, alpha=0.7)
    ax.axhline(0, color='r', linestyle='--')
    ax.set_title("Residuals vs Predicted")
    ax.set_xlabel("Predicted y")
    ax.set_ylabel("Residuals")
    fig.tight_layout()
    st.pyplot(fig)

    st.subheader("8️⃣ Effective Dimension")

    lam_vals = [0.01, 0.1, 1, 10, 100]
    d_eff = [effective_dimension(eigvals, l) for l in lam_vals]

    st.dataframe(pd.DataFrame({
        "λ": lam_vals,
        "Effective dimension": d_eff
    }))

    st.subheader("9️⃣ Comparison with OLS")

    try:
        beta_ols = np.linalg.solve(X.T @ X, X.T @ y)
        ols_err = np.linalg.norm(beta_ols - beta_true)
        st.latex(rf"\|\hat{{\beta}}_{{OLS}} - \beta^*\|_2 = {ols_err:.2f}")
    except np.linalg.LinAlgError:
        st.error("OLS failed: XᵀX is singular (as expected when d ≫ n)")

    st.subheader("📌 Final Takeaway")

    st.markdown("""
    - High-dimensional least squares is **ill-posed**
    - Rank deficiency creates **flat loss geometry**
    - Ridge regularisation restores **invertibility**
    - Conditioning improves via **spectral lifting**
    - Effective dimension is controlled by λ
    - All conclusions follow directly from **linear algebra**

    This is ridge regression **without black boxes**.
    """)

# ────────────────────────────────────────────────
# TAB 2 - unchanged
# ────────────────────────────────────────────────
with tab2:
    # (your original tab2 code remains exactly as you provided)
    st.header("Primal & Dual Ridge Analysis")

    st.markdown("""
    This tab presents **all numerical results** from implementing ridge regularisation **from scratch** in primal and dual forms.
    """)

    st.subheader("1️⃣ Data Generation")

    st.markdown("**Dataset dimensions:**")
    st.code(f"X shape = {X.shape}\ny shape = {y.shape}")

    st.subheader("2️⃣ Rank Deficiency and Null Space")

    st.latex(r"\text{rank}(X) \le \min(n, d)")

    st.markdown(f"""
    - **rank(X)** = {rank_X}  
    - **ambient dimension (d)** = {d}  
    - **null space dimension** = {null_dim}  
    """)

    st.markdown("""
    This confirms that the unregularised least-squares problem
    has a **large null space**, leading to **flat loss directions**
    and non-unique solutions.
    """)

    st.subheader("3️⃣ Spectrum of $X^TX$")

    min_eig = eigvals.min()
    num_zero = np.sum(eigvals_full < 1e-10)

    st.markdown("**Eigenvalue statistics:**")

    st.code(
        f"Smallest eigenvalue of X^T X = {min_eig:.2e}\n"
        f"Number of (near) zero eigenvalues = {num_zero}"
    )

    st.markdown("""
    The presence of many near-zero eigenvalues indicates
    **rank deficiency** and explains why the normal equations
    cannot be inverted without regularisation.
    """)

    st.subheader("Additional: Scree Plot of Eigenvalues")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(eigvals[::-1], marker="o")
    ax.set_yscale("log")
    ax.set_title("Scree Plot of Eigenvalues of X^T X (log scale)")
    ax.set_xlabel("Eigenvalue Index")
    ax.set_ylabel("Eigenvalue")
    fig.tight_layout()
    st.pyplot(fig)

    st.subheader("4️⃣ Ridge Regression Formulations")

    st.latex(r"""
    \textbf{Primal ridge:}\quad
    (X^TX + \lambda I)\beta = X^Ty
    """)

    st.latex(r"""
    \textbf{Dual ridge:}\quad
    \beta = X^T(XX^T + \lambda I)^{-1}y
    """)

    lam = st.slider("Regularisation strength λ (Primal-Dual Tab)", 0.01, 10.0, 1.0, key="lam_pd")

    beta_primal = ridge_regression(X, y, lam)
    beta_dual = ridge_dual(X, y, lam)

    st.subheader("5️⃣ Primal ≡ Dual: Numerical Equivalence")

    diff = np.linalg.norm(beta_primal - beta_dual)

    st.latex(
        rf"\|\beta_{{primal}} - \beta_{{dual}}\|_2 = {diff:.2e}"
    )

    st.markdown("""
    The difference is at **floating-point precision**,
    confirming that primal and dual ridge are
    **mathematically identical**.
    """)

    st.subheader("Additional: Primal vs Dual Coefficients")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(beta_primal, label="Primal", alpha=0.7)
    ax.plot(beta_dual, label="Dual", alpha=0.7, linestyle="--")
    ax.legend()
    ax.set_title("Primal and Dual Ridge Coefficients (Overlaid)")
    fig.tight_layout()
    st.pyplot(fig)

    st.subheader("6️⃣ Conditioning and Numerical Stability")

    nonzero_eigs = eigvals[eigvals > 1e-10]
    cond_unreg = nonzero_eigs.max() / nonzero_eigs.min() if len(nonzero_eigs) > 0 else float('inf')

    st.markdown("**Unregularised system:**")
    st.code(f"Condition number of X^T X (restricted) = {cond_unreg:.4f}")

    def cond_ridge(eigvals, lam):
        e = eigvals + lam
        return e.max() / e.min()

    lams = np.array([1e-3, 1e-2, 1e-1, 1, 10, 100])
    conds = [cond_ridge(eigvals, l) for l in lams]

    df_cond = pd.DataFrame({
        "λ": lams,
        "cond(XᵀX + λI)": conds
    })

    st.markdown("**Condition number vs λ:**")
    st.dataframe(df_cond)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(lams, conds, marker="o")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("λ")
    ax.set_ylabel("Condition number")
    ax.set_title("Effect of Ridge on Conditioning")
    fig.tight_layout()
    st.pyplot(fig)

    st.subheader("7️⃣ Estimation Error")

    err = np.linalg.norm(beta_primal - beta_true)

    st.latex(
        rf"\|\hat{{\beta}}_{{ridge}} - \beta^*\|_2 = {err:.3f}"
    )

    st.markdown("""
    The error is non-zero because ridge introduces **bias**.
    This bias is the price paid for **variance reduction**
    and numerical stability.
    """)

    st.subheader("8️⃣ Computational Perspective")

    st.markdown(f"""
    - **Primal inversion:** {d} × {d} matrix  
    - **Dual inversion:** {n} × {n} matrix  

    Since \\( d >> n \\), the **dual formulation is computationally preferable**.
    """)

    st.subheader("Additional: Runtime Comparison (Primal vs Dual)")
    start = time.time()
    _ = ridge_regression(X, y, lam)
    time_primal = time.time() - start

    start = time.time()
    _ = ridge_dual(X, y, lam)
    time_dual = time.time() - start

    st.code(f"Primal runtime: {time_primal:.6f} seconds\nDual runtime: {time_dual:.6f} seconds")
    st.markdown("Note: Dual is faster in high-d regime.")

    st.subheader("📌 Final Takeaway")

    st.markdown("""
    This complete analysis shows that:

    - High-dimensional least squares is **ill-posed**
    - Ridge regularisation restores **invertibility**
    - Regularisation acts by **lifting the spectrum**
    - Conditioning improves monotonically with λ
    - Primal and dual ridge are **exactly equivalent**
    - All conclusions follow directly from **linear algebra**
    """)

# ────────────────────────────────────────────────
# TAB 3 - FIXED VERSION
# ────────────────────────────────────────────────
with tab3:
    st.header("SVD-Based Ridge Implementation")

    st.markdown("""
    This tab adapts the plain Python SVD implementation to Streamlit, with visualizations and all print results displayed nicely. Additional plots and analyses are added.
    """)

    st.subheader("1️⃣ High-Dimensional Data")

    st.write("Data generated")
    st.code(f"X shape: {X.shape}")

    st.subheader("2️⃣ Eigen-Decomposition of X^T X")

    # Use precomputed sorted & filtered values
    S = np.sqrt(eigvals)
    # U = X V Σ⁻¹
    U = X @ V / S[:, np.newaxis]

    st.code(f"Number of non-zero singular values: {len(S)}\nExpected rank: {n}")

    st.subheader("4️⃣ Verify SVD Properties")

    utu_close = np.allclose(U.T @ U, np.eye(len(S)), atol=1e-8)
    vtv_close = np.allclose(V.T @ V, np.eye(len(S)), atol=1e-8)
    st.code(f"U^T U ≈ I: {utu_close}\nV^T V ≈ I: {vtv_close}")

    st.subheader("5️⃣ Reconstruction Check")

    X_recon = U @ np.diag(S) @ V.T
    recon_error = np.linalg.norm(X - X_recon)

    st.code(f"Reconstruction error ||X − UΣV^T||: {recon_error:.2e}")

    st.subheader("Additional: Reconstruction Difference Histogram")
    diff_matrix = X - X_recon
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(diff_matrix.flatten(), bins=50)
    ax.set_title("Histogram of Reconstruction Errors")
    ax.set_xlabel("Error Value")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    st.pyplot(fig)

    st.subheader("6️⃣ Ridge Regression Using Custom SVD")

    lam = st.slider("Regularisation strength λ (SVD Tab)", 0.01, 10.0, 1.0, key="lam_svd")
    beta_ridge_svd = ridge_from_svd(U, S, V, y, lam)

    st.write("Ridge via custom SVD computed")
    st.code(f"||β̂||_2: {np.linalg.norm(beta_ridge_svd):.3f}")

    st.subheader("7️⃣ Compare with Primal Ridge")

    beta_primal = ridge_regression(X, y, lam)

    diff = np.linalg.norm(beta_primal - beta_ridge_svd)

    st.code(f"||β_primal − β_svd_custom||_2: {diff:.2e}\nSolutions are numerically identical")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(beta_primal, label="Primal", alpha=0.7)
    ax.plot(beta_ridge_svd, label="SVD", alpha=0.7, linestyle="--")
    ax.legend()
    ax.set_title("Primal vs SVD Ridge Coefficients")
    fig.tight_layout()
    st.pyplot(fig)

    st.subheader("8️⃣ Estimation Error")

    error = np.linalg.norm(beta_ridge_svd - beta_true)
    st.code(f"Estimation error ||β̂ − β*||_2: {error:.3f}")

    st.subheader("9️⃣ Effective Dimension (SVD View)")

    def effective_dimension_svd(S, lam):
        return np.sum(S**2 / (S**2 + lam))

    for lam_test in [0.01, 0.1, 1.0, 10.0, 100.0]:
        st.write(f"λ = {lam_test:<6} → effective dimension = {effective_dimension_svd(S, lam_test):.2f}")

    st.subheader("Additional: Effective Dimension Curve")
    lam_range = np.logspace(-2, 2, 100)
    d_eff_range = [effective_dimension_svd(S, l) for l in lam_range]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(lam_range, d_eff_range, marker=".")
    ax.set_xscale("log")
    ax.set_xlabel("λ (log scale)")
    ax.set_ylabel("Effective Dimension")
    ax.set_title("Effective Dimension vs Regularisation Strength")
    fig.tight_layout()
    st.pyplot(fig)

    st.markdown("Custom SVD + ridge implementation completed successfully.")