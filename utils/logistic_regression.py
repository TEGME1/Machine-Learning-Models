import numpy as np


def sigmoid(z):
    g_z = 1 / (1 + np.exp(-z))

    return g_z


def logistic_cost(X, y, w, b):
    m, n = X.shape
    total_cost = 0
    for i in range(m):
        f_wb = sigmoid(np.dot(w, X[i]) + b)
        total_cost = total_cost + (y[i] * -np.log(f_wb) + (1 - y[i]) * -np.log(1 - f_wb))

    total_cost = total_cost / m

    return total_cost


def logistic_cost_gradients(X, y, w, b):
    dj_dw = np.zeros_like(w)
    dj_db = 0.0

    m, n = X.shape

    for i in range(m):
        z = np.dot(w, X[i]) + b
        g_z = sigmoid(z)
        f_x = g_z
        error = f_x - y[i]
        dj_dw = dj_dw + error * X[i]
        dj_db = dj_db + error

    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db
