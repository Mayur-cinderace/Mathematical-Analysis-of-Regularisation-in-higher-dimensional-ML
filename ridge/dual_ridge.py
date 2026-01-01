import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="High-Dimensional Ridge Regularisation",
    layout="wide"
)

st.title("📐 High-Dimensional Ridge Regularisation")
st.subheader("Complete Linear Algebra Analysis (Primal & Dual)")

st.markdown("""
This page presents **all numerical results** obtained from
implementing ridge regularisation **from scratch** in the
high-dimensional regime:

\\[
d >> n
\\]

Every quantity shown below corresponds directly to the
computations performed in the Python implementation.
""")

# ============================================================
# 1. DATA GENERATION
# ============================================================

st.header("1️⃣ Data Generation")

np.random.seed(42)

n = 60
d = 500
s = 8
sigma = 0.1

X = np.random.randn(n, d)
beta_true = np.zeros(d)
beta_true[:s] = np.random.randn(s)
y = X @ beta_true + sigma * np.random.randn(n)

st.markdown("**Dataset dimensions:**")
st.code(f"X shape = {X.shape}\ny shape = {y.shape}")

# ============================================================
# 2. RANK & NULL SPACE
# ============================================================

st.header("2️⃣ Rank Deficiency and Null Space")

rank_X = np.linalg.matrix_rank(X)
null_dim = d - rank_X

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

# ============================================================
# 3. SPECTRAL ANALYSIS OF XᵀX
# ============================================================

st.header("3️⃣ Spectrum of $X^TX$")

XtX = X.T @ X
eigvals = np.linalg.eigvalsh(XtX)

min_eig = eigvals.min()
num_zero = np.sum(eigvals < 1e-10)

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

# ============================================================
# 4. RIDGE DEFINITIONS
# ============================================================

st.header("4️⃣ Ridge Regression Formulations")

st.latex(r"""
\textbf{Primal ridge:}\quad
(X^TX + \lambda I)\beta = X^Ty
""")

st.latex(r"""
\textbf{Dual ridge:}\quad
\beta = X^T(XX^T + \lambda I)^{-1}y
""")

# ============================================================
# 5. PRIMAL & DUAL IMPLEMENTATIONS
# ============================================================

def ridge_primal(X, y, lam):
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)

def ridge_dual(X, y, lam):
    alpha = np.linalg.solve(X @ X.T + lam * np.eye(X.shape[0]), y)
    return X.T @ alpha

lam = st.slider("Regularisation strength λ", 0.01, 10.0, 1.0)

beta_primal = ridge_primal(X, y, lam)
beta_dual = ridge_dual(X, y, lam)

# ============================================================
# 6. PRIMAL ≡ DUAL (NUMERICAL)
# ============================================================

st.header("5️⃣ Primal ≡ Dual: Numerical Equivalence")

diff = np.linalg.norm(beta_primal - beta_dual)

st.latex(
    rf"\|\beta_{{primal}} - \beta_{{dual}}\|_2 = {diff:.2e}"
)

st.markdown("""
The difference is at **floating-point precision**,
confirming that primal and dual ridge are
**mathematically identical**.
""")

# ============================================================
# 7. CONDITION NUMBER ANALYSIS
# ============================================================

st.header("6️⃣ Conditioning and Numerical Stability")

nonzero_eigs = eigvals[eigvals > 1e-10]
cond_unreg = nonzero_eigs.max() / nonzero_eigs.min()

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

fig, ax = plt.subplots()
ax.plot(lams, conds, marker="o")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("λ")
ax.set_ylabel("Condition number")
ax.set_title("Effect of Ridge on Conditioning")

st.pyplot(fig)

# ============================================================
# 8. ESTIMATION ERROR
# ============================================================

st.header("7️⃣ Estimation Error")

err = np.linalg.norm(beta_primal - beta_true)

st.latex(
    rf"\|\hat{{\beta}}_{{ridge}} - \beta^*\|_2 = {err:.3f}"
)

st.markdown("""
The error is non-zero because ridge introduces **bias**.
This bias is the price paid for **variance reduction**
and numerical stability.
""")

# ============================================================
# 9. COMPUTATIONAL COST
# ============================================================

st.header("8️⃣ Computational Perspective")

st.markdown(f"""
- **Primal inversion:** {d} × {d} matrix  
- **Dual inversion:** {n} × {n} matrix  

Since \\( d >> n \\), the **dual formulation is computationally preferable**.
""")

# ============================================================
# 10. FINAL TAKEAWAY
# ============================================================

st.header("📌 Final Takeaway")

st.markdown("""
This complete analysis shows that:

- High-dimensional least squares is **ill-posed**
- Ridge regularisation restores **invertibility**
- Regularisation acts by **lifting the spectrum**
- Conditioning improves monotonically with λ
- Primal and dual ridge are **exactly equivalent**
- All conclusions follow directly from **linear algebra**
""")
