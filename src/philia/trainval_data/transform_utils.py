import numpy as np


def format_dist(x):
    x = np.stack(x)
    return x

def scale_dist(x):
    x = (x/30.0 - 0.5)*6.0
    return x

def format_angles(x):
    x = np.stack(x)  # [9, 2]
    x[0, 0] = x[-1, 0]  # bring second-to-last to first nan slot
    x = x[:-1]
    return x

def scale_angles(x):
    x = np.deg2rad(x)
    return x
