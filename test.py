import numpy as np
from utils import squared_cost_function, squared_cost_gradients, gradient_descent, z_score_normalization
import matplotlib.pyplot as plt

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]], dtype='float')
y_train = np.array([460, 232, 178])

initial_w = np.zeros_like(X_train[0])
initial_b = 0

mu, sigma, X_train = z_score_normalization(X_train)

print(X_train)
# print(squared_cost_gradients(X_train, y_train, initial_w, initial_b))
w, b, cost_history = gradient_descent(X_train, y_train, initial_w, initial_b, squared_cost_function,
                                      squared_cost_gradients, 1000,
                                      0.2455)
for i in range(len(X_train)):
    print(f'{np.dot(w, X_train[i]) + b} -- {y_train[i]}')

#
# print(X_scaled)
