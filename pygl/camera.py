import math

import numpy as np

from . import transform
from .mesh import Mesh


def perspective(fov, aspect, near, far):
    num = 1.0 / np.tan(fov * 0.5)
    idiff = 1.0 / (near - far)
    return np.array([
        [num/aspect, 0, 0, 0],
        [0, num, 0, 0],
        [0, 0, (far + near) * idiff, 2*far*near*idiff],
        [0, 0, -1, 0]
    ], dtype=np.float32)

def orthographic(left, right, bottom, top, near, far):
    return np.array([
        [2/(right - left), 0, 0, -(right + left)/(right - left)],
        [0, 2/(top - bottom), 0, -(top + bottom)/(top - bottom)],
        [0, 0, -2/(far - near), -(far + near)/(far - near)],
        [0, 0, 0, 1]
    ], dtype=np.float32)

class Camera(object):

    def __init__(self, screen_size, near=0.05, far=300.0, fov=50.0):
        self.__T_view = np.eye(4, dtype=np.float32)
        self.__T_proj = np.eye(4, dtype=np.float32)
        self.__near = near
        self.__far  = far
        self.__dirty = True
        self.__size = screen_size
        self.__fov = fov

    def __recompute_matrices(self):
        self.__T_proj = perspective(math.radians(self.fov), self.aspect_ratio, self.near, self.far)
        self.__dirty = False

    def look_at(self, target, eye=None, up=[0, 1, 0]):
        def compute_lookat(pts):
            pt_min = np.min(pts, axis=0)
            pt_max = np.max(pts, axis=0)
            center = (pt_min + pt_max) * 0.5
            r   = np.linalg.norm(pts - center, axis=-1).max()
            d   = r / np.tan(0.5 * math.radians(self.fov))
            eye = center + np.array([0, d*0.707, d*0.707])
            self.__T_view = transform.look_at(eye, center, up)

        if isinstance(target, Mesh):
            compute_lookat(target.vertices)
        elif isinstance(target, np.ndarray) and target.ndim > 1:
            compute_lookat(target.reshape((-1, 3)))
        else:
            if eye is None:
                eye = self.position
            self.__T_view = transform.look_at(eye, target, up)

    @property
    def position(self):
        return self.__T_view[:, 3]

    @property
    def aspect_ratio(self):
        # screen_width / screen_height
        return self.__size[1] / self.__size[0]

    @property
    def VP(self):
        if self.__dirty:
            self.__recompute_matrices()
        return self.__T_proj @ self.__T_view

    @property
    def P(self):
        if self.__dirty:
            self.__recompute_matrices()
        return self.__T_proj

    @property
    def V(self):
        return self.__T_view

    @V.setter
    def V(self, T):
        self.__T_view = T

    @property
    def screen_size(self):
        return self.__size
    @property
    def screen_width(self):
        return self.__size[1]
    @property
    def screen_height(self):
        return self.__size[0]

    @screen_size.setter
    def screen_size(self, x):
        self.__dirty = True
        self.__size = x

    @property
    def far(self):
        return self.__far

    @far.setter
    def far(self, x):
        self.__dirty = True
        self.__far = x

    @property
    def near(self):
        return self.__near

    @near.setter
    def near(self, x):
        self.__dirty = True
        self.__near = x

    @property
    def fov(self):
        return self.__fov
    
    @fov.setter
    def fov(self, x):
        self.__dirty = True
        self.__fov = x
