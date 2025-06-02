import numpy as np


def combine_dict(list_of_dict):
    new_dict = {}
    for k in list_of_dict[0].keys():
        new_dict[k] = np.concatenate([d[k] for d in list_of_dict], axis=0)
    return new_dict
