import numpy as np
import pandas


def h(a, b, x):
    result = a + b * x
    return result


def func_loss(predict_value, true_value):
    quantity_values = len(true_value)
    result = np.sum((predict_value - true_value) ** 2) / (2 * quantity_values)
    return result
