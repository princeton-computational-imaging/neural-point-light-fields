import numpy as np

def invert_transformation(rot, t):
    t = np.matmul(-rot.T, t)
    inv_translation = np.concatenate([rot.T, t[:, None]], axis=1)
    return np.concatenate([inv_translation, np.array([[0., 0., 0., 1.]])])

def roty_matrix(roty):
    c = np.cos(roty)
    s = np.sin(roty)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rotz_matrix(roty):
    c = np.cos(roty)
    s = np.sin(roty)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])