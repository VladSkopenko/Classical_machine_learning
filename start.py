import numpy as np


def h(X, w):
    return np.dot(X, w)


def loss_function(X, y, w):
    return np.square(h(X, w) - y).sum() / (2 * len(X))


def gradient_step(X, y, w, learning_rate):
    m = len(y)
    grad = (X.T @ (h(X, w) - y)) / m
    w -= learning_rate * grad
    return w


def gradient(X, y, learning_rate, num_iter, eps):
    ones = np.ones((X.shape[0], 1))
    X = np.hstack((ones, X))

    w = np.zeros(X.shape[1])

    loss = loss_function(X, y, w)
    loss_history = [loss]

    for _ in range(num_iter):
        w = gradient_step(X, y, w, learning_rate)

        loss = loss_function(X, y, w)
        if abs(loss - loss_history[-1]) < eps:
            loss_history.append(loss)
            break

        loss_history.append(loss)

    return w, loss_history
