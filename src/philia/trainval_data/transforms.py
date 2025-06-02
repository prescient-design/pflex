import numpy as np
import torch


PI = 3.141592653589


def deg2rad(x):
    return x*PI/180.0


def rad2deg(x):
    return x*180.0/PI


def scale_dist(x):
    # x = (np.log(x)/3.5 - 0.5)*2.0
    x = x/30.0 - 1.0
    return x

def scale_dist_log(x):
    x = np.log(x) / 2.0 - 1.5
    # x = np.log(x) / 1.2 - 2.0
    return x

def scale_angles(x):
    # x = np.deg2rad(x)
    # x = x/3.14
    x = x/180.0  # to [-1, 1]
    return x

# def scale_angles_flip(x):
#     for i in range(x.shape[0]):
#         for j in range(x.shape[1]):
#             if x[i,j] <0:
#                 x[i,j] = x[i,j] + 360
#     x = x/180.0
#     return x


def scale_phi(x):
    # x = x/180.0 + 0.5  # to [-1, 1] + 0.5
    x = x / 180.0
    # x = np.log((1 + x) / 180) / 5 + 1
    return x


def scale_psi(x):
    x = x / 180.0
    # x = x/180.0 - 0.75  # to [-1, 1] - 0.75
    # x = np.log((1 + x) / 180) / 5 + 1
    return x


def scale_angles_separately(x):
    # x ~ [9, 3]
    x[:, 0] = scale_phi(x[:, 0])
    x[:, 1] = scale_psi(x[:, 1])
    return x


def inverse_scale_dist(x):
    # return np.exp((0.5*x + 0.5)*3.5)
    return 30.0*(x + 1.0)
    # return np.exp(2*(x+1.5))

def inverse_scale_dist_log(x):
    return np.exp(2*(x+1.5))
    # return np.exp(1.2*(x+2.0))

def inverse_scale_dist_std(x):
    # return np.exp((0.5*x + 0.5)*3.5)
    # return x*15.0
    return x*30.0
    # return np.exp(2*(x+1.5))

def inverse_scale_dist_std_log(x):
    return np.exp(2*(x+1.5))
    # return np.exp(1.2*(x+2.0))

def inverse_scale_angles(x):
    return x*180.0

# def inverse_scale_angles_flip(x):
#     x = x * 180.0
#     for i in range(x.shape[0]):
#         for j in range(x.shape[1]):
#             if x[i,j] > 180:
#                 x[i,j] = x[i,j] - 360
#     return x

def inverse_scale_angles_std(x):
    return x*180.0
    # return 180 * np.exp(5 * (x - 1))

def inverse_scale_phi(x):
    # x = (x - 0.5)*180.0
    x = x * 180.0
    # x = 180 * np.exp(5 * (x - 1)) - 1
    
    return x


def inverse_scale_psi(x):
    # x = (x + 0.75)*180.0
    x = x * 180.0
    # x = 180 * np.exp(5 * (x - 1)) - 1
    return x


def inverse_scale_angles_separately(x):
    x[:, 0] = inverse_scale_phi(x[:, 0])
    x[:, 1] = inverse_scale_psi(x[:, 1])
    return x
