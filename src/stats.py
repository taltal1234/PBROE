"""The statistics used in the paper, written out by hand so it is clear what each number is."""
import numpy as np
import pandas as pd
import config as C


def newey_west(x, lags=C.NW_LAGS):
    """Mean of a series, its Newey West t statistic, and the number of observations."""
    x = pd.Series(x).dropna().to_numpy(float)
    n = len(x)
    e = x - x.mean()
    s = e @ e / n
    for lag in range(1, lags + 1):
        s += 2 * (1 - lag / (lags + 1)) * (e[lag:] @ e[:-lag]) / n
    return x.mean(), x.mean() / np.sqrt(s / n), n


def ols_newey_west(y, X, lags=C.NW_LAGS):
    """OLS with Newey West standard errors.

    X comes without a constant; one is added in front, so coefficient 0 is the alpha.
    Returns the coefficients and their t statistics.
    """
    X = np.column_stack([np.ones(len(X)), X])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    u = y - X @ b
    n = len(y)
    S = (X * u[:, None]).T @ (X * u[:, None]) / n
    for lag in range(1, lags + 1):
        G = (X[lag:] * u[lag:, None]).T @ (X[:-lag] * u[:-lag, None]) / n
        S += (1 - lag / (lags + 1)) * (G + G.T)
    Q = np.linalg.inv(X.T @ X / n)
    V = Q @ S @ Q / n
    return b, b / np.sqrt(np.diag(V))


def winsorize(s, lo=0.01, hi=0.99):
    return s.clip(s.quantile(lo), s.quantile(hi))


def zscore(s):
    return (s - s.mean()) / s.std()


def ols_residuals(df, y_col, x_cols, dummies=None):
    """Residuals of y_col on x_cols within one cross section, plus the slope on the first x.

    With dummies, every group in that column gets its own intercept (so no separate constant).
    """
    X = df[x_cols].to_numpy(float)
    if dummies is not None:
        X = np.column_stack([X, pd.get_dummies(df[dummies], drop_first=False).to_numpy(float)])
    else:
        X = np.column_stack([X, np.ones(len(df))])
    y = df[y_col].to_numpy(float)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return y - X @ beta, beta[0]


def hit_and_chance(cheap, beat):
    """How often 'cheap' and 'beat the average' agree, and how often they would agree by chance.

    If a share p of stocks is cheap and a share q beats the average, two independent coin
    flips agree with probability p q + (1 - p)(1 - q). That is the bar the hit rate has to clear.
    """
    p, q = cheap.mean(), beat.mean()
    return {"hit": (cheap == beat).mean(), "base": p * q + (1 - p) * (1 - q)}
