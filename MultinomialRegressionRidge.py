import numpy as np
import pandas as pd
from itertools import combinations_with_replacement as cwr, chain
import matplotlib.pyplot as plt

class Poly_reg:
    def __init__(self, n_features, degree, lambda_reg=1.0):
        self.n_features = n_features
        self.degree = degree
        self.lambda_reg = float(lambda_reg)

        self.monomials = list(chain.from_iterable(
            cwr(range(self.n_features), d) for d in range(self.degree + 1)
        ))

        self.monomial_indices = np.full((len(self.monomials), self.degree), -1, dtype=np.int64)
        for i, tup in enumerate(self.monomials):
            self.monomial_indices[i, :len(tup)] = tup

        self.W = np.zeros(len(self.monomials), dtype=float)

    def __poly_transform(self, X):
        X = np.asarray(X, dtype=float)

        idx = self.monomial_indices.copy()
        idx[idx == -1] = 0  # safe index for padding

        X_tmp = X[:, idx]  # (n, n_monomials, degree)
        X_tmp[:, self.monomial_indices == -1] = 1.0
        return np.prod(X_tmp, axis=2)  # (n, n_monomials)

    def fit(self, X, Y):
        X_poly = self.__poly_transform(X)
        Y = np.asarray(Y, dtype=float).reshape(-1)

        A = X_poly.T @ X_poly
        if self.lambda_reg != 0:
            R = np.eye(A.shape[0], dtype=float)
            R[0, 0] = 0.0  # do NOT penalize intercept (constant term)
            A = A + self.lambda_reg * R

        b = X_poly.T @ Y
        self.W = np.linalg.solve(A, b)

    def predict(self, X):
        return self.__poly_transform(X) @ self.W


def repeated_cross_validation(X, Y, lambda_list, k=5, repeats=5, seed=42, degree=3):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float).reshape(-1)

    n = X.shape[0]
    best_lambda, best_rmse = None, np.inf
    rmse_avg, rmse_std = [], []

    for lam in lambda_list:
        rep_scores = []
        for r in range(repeats):
            rng = np.random.default_rng(seed + 1009 * r)
            indices = rng.permutation(n)
            folds = np.array_split(indices, k)

            fold_scores = []
            for i in range(k):
                val_idx = folds[i]
                train_idx = np.concatenate([f for j, f in enumerate(folds) if j != i])

                model = Poly_reg(X.shape[1], degree, lam)
                model.fit(X[train_idx], Y[train_idx])
                pred = model.predict(X[val_idx])

                fold_scores.append(np.sqrt(np.mean((Y[val_idx] - pred) ** 2)))

            rep_scores.append(np.mean(fold_scores))

        m, s = float(np.mean(rep_scores)), float(np.std(rep_scores, ddof=1) if repeats > 1 else 0.0)
        rmse_avg.append(m); rmse_std.append(s)

        if m < best_rmse:
            best_rmse, best_lambda = m, lam

    return best_lambda, np.array(rmse_avg), np.array(rmse_std)


if __name__ == "__main__":
    data = pd.read_csv("data_hw1_2025.csv").to_numpy(dtype=float)
    X, Y = data[:, :-1], data[:, -1]
    seed = int  (np.random.rand()*1000)
    print(f"seed = {seed}")

    lambdas = [0, 0.001, 0.01, 0.05, 0.1, 0.5, 1, 2, 3, 4, 4.5, 5, 5.5, 6, 6.5, 7, 10]
    best_lambda, rmse_avg, rmse_std = repeated_cross_validation(
        X, Y, lambdas, k=5, repeats=10, seed=seed, degree=3
    )

    print("best_lambda =", best_lambda)

    plt.plot(lambdas, rmse_avg, marker="o")
    plt.fill_between(lambdas, rmse_avg - rmse_std, rmse_avg + rmse_std, alpha=0.2)
    plt.xlabel("Lambda")
    plt.ylabel("Average CV RMSE")
    plt.title("Repeated k-Fold CV for Ridge Regression")
    plt.grid(True)
    from pathlib import Path

    out = Path.home() / "Desktop/Slike" / f"cv_lambda_seed{seed}_k5_R10_deg3.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print("Saved plot to:", out)

    plt.show()
