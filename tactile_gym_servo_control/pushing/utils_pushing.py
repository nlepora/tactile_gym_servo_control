import pickle
from collections import Mapping

import numpy as np

from cri.transforms import euler2mat, mat2euler, euler2quat, quat2euler, transform, inv_transform


# Persistable namespace
class Namespace(Mapping):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __repr__(self):
        return self.__dict__.__repr__()

    def __iter__(self):
        return iter(self.__dict__)

    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, item, value):
        self.__dict__[item] = value

    def __len__(self):
        return len(self.__dict__)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.__getstate__(), f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            tmp = pickle.load(f)
        self.__setstate__(tmp)


# Helper functions for transforming poses expressed in Euler notation
def transform_euler(pose_a, frame_b_a, axes='sxyz', use_quat=True):
    if use_quat:
        return quat2euler(transform(euler2quat(pose_a, axes),
                                    euler2quat(frame_b_a, axes)), axes)
    return mat2euler(np.dot(np.linalg.pinv(euler2mat(frame_b_a, axes)),
                            euler2mat(pose_a, axes)), axes)

def inv_transform_euler(pose_b, frame_b_a, axes='sxyz', use_quat=True):
    if use_quat:
        return quat2euler(inv_transform(euler2quat(pose_b, axes),
                                        euler2quat(frame_b_a, axes)), axes)
    return mat2euler(np.dot(euler2mat(frame_b_a, axes),
                            euler2mat(pose_b, axes)), axes)
