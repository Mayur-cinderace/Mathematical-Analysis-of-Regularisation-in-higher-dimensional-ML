# 📐 Mathematical Analysis of Regularisation in Higher-Dimensional ML

A comprehensive linear algebra–based experimental study of regularisation techniques in high-dimensional machine learning, implemented entirely from scratch.

## 📋 Overview

This project demonstrates **ridge (L2) and LASSO (L1) regularisation** in the high-dimensional regime where $d \gg n$. All implementations use fundamental linear algebra principles, without relying on black-box libraries.

**Key Focus:**
- Geometry of regularisation constraints
- Spectral analysis and conditioning
- Primal vs dual formulations
- SVD-based implementations
- Effective dimension analysis

## 📁 Project Structure

```
.
├── ridge/                          # Ridge regression implementations
│   ├── ridge_regul.py             # Complete linear algebra study
│   ├── ridge_regul.ipynb          # Jupyter notebook version
│   ├── dual_ridge.py              # Primal & dual formulations
│   ├── dual_ridge.ipynb           # Notebook: primal ≡ dual analysis
│   ├── svd_ridge.py               # SVD-based ridge regression
│   └── svd_ridge.ipynb            # Notebook: SVD implementation
│
├── lasso/                          # LASSO (L1 regularisation)
│   ├── l1.py                      # LASSO via coordinate descent
│   ├── l1.ipynb                   # Notebook: coordinate descent solver
│   ├── l1_vis.py                  # L1 vs L2 geometry visualization
│   └── l1_vis.ipynb               # Notebook: constraint geometry
│
├── ridge_app.py                    # Interactive Streamlit dashboard
├── eval.py                         # Model evaluation utilities
├── spectra.py                      # Spectral analysis tools
├── bert_test.py                    # BERT-based evaluation
├── complexity_results.csv          # Benchmarking results
├── README.md                       # This file
└── .gitignore                      # Git ignore rules
```

## 🚀 Getting Started

### Requirements
```
numpy
pandas
matplotlib
streamlit
scikit-learn
```

### Installation
```bash
git clone https://github.com/Mayur-cinderace/Mathematical-Analysis-of-Regularisation-in-higher-dimensional-ML.git
cd Mathematical-Analysis-of-Regularisation-in-higher-dimensional-ML
pip install numpy pandas matplotlib streamlit scikit-learn
```

## 📚 Module Descriptions

### Ridge Regression (`ridge/`)

#### `ridge_regul.py`
Complete experimental study of high-dimensional ridge regression:
- Data generation with rank deficiency ($d >> n$)
- PCA visualization
- Rank analysis and null space dimension
- Loss geometry via eigendecomposition
- Ridge solution computation
- Effective dimension analysis
- Comparison with OLS

**Run:**
```bash
python ridge/ridge_regul.py
```

#### `dual_ridge.py`
Primal and dual implementations demonstrating equivalence:
- Primal formulation: $(X^TX + \lambda I)\beta = X^Ty$
- Dual formulation: $\beta = X^T(XX^T + \lambda I)^{-1}y$
- Condition number analysis
- Numerical equivalence verification
- Computational cost comparison

**Run:**
```bash
python ridge/dual_ridge.py
```

#### `svd_ridge.py`
Ridge regression via explicit SVD decomposition:
- Eigen-decomposition of $X^TX$
- Singular value filtering
- Ridge shrinkage operator: $\sigma_i / (\sigma_i^2 + \lambda)$
- Effective dimension from SVD perspective
- Reconstruction error analysis

**Run:**
```bash
python ridge/svd_ridge.py
```

### LASSO Regression (`lasso/`)

#### `l1.py`
LASSO via coordinate descent (from scratch):
- **Soft-thresholding operator:** $S(z, \lambda) = \text{sign}(z)(\max(|z| - \lambda, 0))$
- Coordinate descent solver
- Sparsity analysis
- Estimation error computation
- Objective: $\frac{1}{2}\|y - X\beta\|_2^2 + \lambda\|\beta\|_1$

**Run:**
```bash
python lasso/l1.py
```

#### `l1_vis.py`
Geometric comparison of L1 vs L2 constraints:
- Level sets of squared loss function
- L1 constraint boundary (diamond shape)
- L2 constraint boundary (circle)
- Visualization of how L1 promotes sparsity

**Run:**
```bash
python lasso/l1_vis.py
```

### Interactive Dashboard

#### `ridge_app.py`
Streamlit-based interactive exploration of ridge regression:
- **Tab 1:** Basic Ridge Report
  - Data generation and visualization
  - Rank deficiency analysis
  - Loss geometry via eigenvalues
  - Ridge solution and error metrics
  
- **Tab 2:** Primal & Dual Analysis
  - Primal and dual implementations
  - Numerical equivalence demonstration
  - Conditioning analysis over $\lambda$ range
  - Effective dimension curves
  
- **Tab 3:** SVD Implementation
  - SVD decomposition verification
  - SVD properties (orthonormality)
  - Reconstruction error
  - Primal vs SVD solution comparison

**Run:**
```bash
streamlit run ridge_app.py
```

Then open your browser to `http://localhost:8501`

### Utilities

#### `eval.py`
Model evaluation functions and metrics.

#### `spectra.py`
Spectral analysis and eigenvalue computations.

#### `bert_test.py`
BERT-based evaluation utilities.

#### `complexity_results.csv`
Benchmarking results comparing algorithmic complexity across methods.

## 🧮 Key Mathematical Concepts

### Ridge Regularisation
Objective:
$$\min_\beta \quad \frac{1}{2}\|y - X\beta\|_2^2 + \lambda\|\beta\|_2^2$$

Normal equations:
$$(X^TX + \lambda I)\beta = X^Ty$$

**Key insight:** Adding $\lambda I$ **lifts the spectrum** away from zero, restoring invertibility.

### Effective Dimension
$$d_{\text{eff}}(\lambda) = \sum_{i=1}^r \frac{\sigma_i^2}{\sigma_i^2 + \lambda}$$

where $\sigma_i$ are singular values. This measures the number of "active" directions in the solution.

### LASSO via Coordinate Descent
Update rule for coefficient $j$:
$$\beta_j^{(t+1)} = \frac{S(\rho_j, \lambda)}{\|X_{\cdot j}\|_2^2}$$

where $\rho_j = X_{\cdot j}^T(y - X\beta^{(t)} + X_{\cdot j}\beta_j^{(t)})$ and $S$ is the soft-threshold operator.

## 📊 Experimental Setup

All code uses a standardised high-dimensional regime:
- **Sample size:** $n = 60$
- **Ambient dimension:** $d = 500$
- **True sparsity:** $s = 8$ (only 8 true features)
- **Noise level:** $\sigma = 0.1$
- **Condition:** $d >> n$ (rank-deficient regime)

This setup clearly demonstrates:
- Ill-posedness of unregularised least squares
- Effectiveness of regularisation
- Geometry of solutions
- Computational benefits of dual formulation

## 🎯 Main Results

### Ridge Regression
1. **Ill-posedness:** Unregularised OLS has infinitely many solutions
2. **Spectral lifting:** Ridge adds $\lambda$ to all eigenvalues, restoring invertibility
3. **Primal ≡ Dual:** Both formulations give identical solutions
4. **Computational:** Dual is faster when $d >> n$ ($n \times n$ vs $d \times d$ inversion)
5. **Effective dimension:** Controlled by regularisation strength $\lambda$

### LASSO (L1 Regularisation)
1. **Sparsity:** L1 penalty promotes exact zeros (not just small values)
2. **Geometry:** Diamond-shaped L1 constraint intersects sparse solutions
3. **Coordinate descent:** Efficient iterative solver using soft-thresholding
4. **Model selection:** Automatic feature selection through sparsity

## 📈 Visualizations

- **PCA projection:** 2D reduction of high-dimensional data
- **Eigenvalue distribution:** Spectrum of $X^TX$ (illustrates rank deficiency)
- **Loss surface:** Contours showing effect of regularisation
- **Condition number curves:** Improvement with increasing $\lambda$
- **L1 vs L2 geometry:** Constraint boundaries and sparser solutions
- **Effective dimension:** Number of active directions vs $\lambda$

## 🔍 Code Quality

- **From-scratch implementations:** No hidden black boxes
- **Detailed comments:** Mathematical notation matched with code
- **Jupyter notebooks:** Interactive exploration with visualizations
- **Comprehensive analysis:** Each script provides full linear algebra breakdown
- **Reproducibility:** Fixed random seed for all experiments

## 💡 Educational Value

This project is ideal for:
- **Understanding regularisation** at a deep linear algebra level
- **Learning about conditioning** and numerical stability
- **Understanding the geometry** of L1 vs L2 penalties
- **Implementing solvers** from first principles
- **Exploring high-dimensional statistics**

## 📝 Files at a Glance

| File | Purpose |
|------|---------|
| `ridge/ridge_regul.py` | Ridge: complete study |
| `ridge/dual_ridge.py` | Ridge: primal vs dual |
| `ridge/svd_ridge.py` | Ridge: SVD implementation |
| `lasso/l1.py` | LASSO: coordinate descent |
| `lasso/l1_vis.py` | LASSO: geometry visualization |
| `ridge_app.py` | Interactive Streamlit dashboard |

## ✨ Key Takeaways

> **Ridge regularisation** without black boxes:
> - High-dimensional least squares is **ill-posed**
> - Ridge **restores invertibility** via spectral lifting
> - **Primal and dual** are mathematically identical
> - **SVD reveals** the geometry of solutions
> - **LASSO promotes sparsity** through L1 geometry

All conclusions follow directly from **linear algebra** 📐

---

