import numpy as np


def gradient_descent(X, y, w, b, cost_function, gradient_function, iterations, alpha, lamda_=0):
    w_out = np.copy(w)
    b_out = np.copy(b)

    cost_history = []

    for i in range(iterations):
        dj_dw, dj_db = gradient_function(X, y, w_out, b_out)
        w_out = w_out - alpha * dj_dw
        b_out = b_out - alpha * dj_db
        cost_history.append(cost_function(X, y, w_out, b_out))

    return w_out, b_out, cost_history
