import numpy as np


def squared_cost_function(X, y, w, b):
    cost = 0.0
    m, n = X.shape

    for i in range(m):
        error = np.dot(w, X[i]) + b - y[i]
        cost = cost + np.square(error)

    cost = cost / (2 * m)

    return cost


def squared_cost_gradients(X, y, w, b):
    dj_dw = np.zeros_like(w)
    dj_db = 0

    m, n = X.shape

    for i in range(m):
        error = np.dot(w, X[i]) + b - y[i]

        dj_dw = dj_dw + error * X[i]
        dj_db = dj_db + error

    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db
