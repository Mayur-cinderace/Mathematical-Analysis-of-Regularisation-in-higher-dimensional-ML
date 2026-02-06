from sklearn.model_selection import train_test_split
import numpy as np

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3)

errors = []
def ridge_primal(X, y, lam):
    return np.linalg.solve(
        X.T @ X + lam * np.eye(X.shape[1]),
        X.T @ y
    )
for lam in lambdas:
    w = ridge_primal(Xtr, ytr, lam)
    err = norm(Xte @ w - yte)
    errors.append(err)

plt.figure()
plt.semilogx(lambdas, errors, marker='o')
plt.xlabel(r'$\lambda$')
plt.ylabel('Test error')
plt.title('Bias–Variance Tradeoff (Ridge)')
plt.grid(True)
plt.show()
