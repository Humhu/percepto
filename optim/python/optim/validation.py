"""Methods for validating models.
"""

import numpy as np

def leave_one_out(train_func, pred_func, X, y):

    if len(X) != len(y):
        raise ValueError('X and y must have same length')

    errs = []
    for i in range(len(X)):
        sub_x = list(X)
        sub_y = list(y)
        x_test = sub_x[i]
        y_test = sub_y[i]
        del sub_x[i]
        del sub_y[i]

        x_test = x_test.reshape(1, -1)

        train_func(sub_x, sub_y)
        err = y_test - pred_func(x_test)
        errs.append(err)
    return errs