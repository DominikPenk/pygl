# This file contains code to render a couple of scenes.
from dataclasses import dataclass
import os

from typing import Dict, List

from pygl import Mesh, Camera
from pygl import transform
from pygl import texture

import numpy as np


def load_object(name:str)->Mesh:
    """Load some of the objects found in the common-3d-test-models repository.
    https://github.com/alecjacobson/common-3d-test-models"""
    default_objects = {
        'armadillo': 'https://github.com/alecjacobson/common-3d-test-models/raw/master/data/armadillo.obj',
        'bunny': 'https://github.com/alecjacobson/common-3d-test-models/raw/master/data/stanford-bunny.obj',
        'cow': 'https://github.com/alecjacobson/common-3d-test-models/blame/master/data/cow.obj',
        'happy_buddha': 'https://github.com/alecjacobson/common-3d-test-models/raw/master/data/happy.obj',
        'homer': 'https://github.com/alecjacobson/common-3d-test-models/blame/master/data/homer.obj',
        'rocker-arm': 'https://github.com/alecjacobson/common-3d-test-models/blame/master/data/rocker-arm.obj',
        'spot': 'https://github.com/alecjacobson/common-3d-test-models/blame/master/data/spot.obj',
        'suzanne': 'https://github.com/alecjacobson/common-3d-test-models/blame/master/data/suzanne.obj',
        'teapot': 'https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/teapot.obj',
        'dragon': 'https://github.com/alecjacobson/common-3d-test-models/raw/master/data/xyzrgb_dragon.obj'
    }
    if name not in default_objects:
        raise ValueError('Unknown object: {}'.format(name))

    object_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'objects')
    if not os.path.exists(object_dir):
        os.makedirs(object_dir)

    # Check if the object is already downloaded.
    object_path = os.path.join(object_dir, name + '.obj')
    object_path = os.path.abspath(object_path)
    if not os.path.exists(object_path):
        # Download the object.
        import requests
        print(f'Downloading object: {name} (Destination: {object_path})')
        r = requests.get(default_objects[name])
        with open(object_path, 'wb') as f:
            f.write(r.content)
    
    return Mesh.load(object_path)

@dataclass
class InstanceInfo:
    object: str
    ModelMatrix: np.ndarray = np.identity(4, np.float32)
    kd: np.ndarray = np.array([0.7, 0.7, 0.7], dtype=np.float32)
    _kd_map: texture.Texture2D = None
    ks: np.ndarray = np.array([0.2, 0.2, 0.2], dtype=np.float32)
    _ks_map: texture.Texture2D = None
    shininess: float = 128.0
    alpha: float = 1.0
    _alpha_map: texture.Texture2D = None

    @property
    def kd_map(self):
        if self._kd_map is None:
            self._kd_map = texture.get_default_texture_2d()
        return self._kd_map
    @kd_map.setter
    def kd_map(self, value):
        self._kd_map = texture.as_texture_2d(value)
    
    @property
    def ks_map(self):
        if self._ks_map is None:
            self._ks_map = texture.get_default_texture_2d()
        return self._ks_map
    @ks_map.setter
    def ks_map(self, value):
        self._ks_map = texture.as_texture_2d(value)
    
    @property
    def alpha_map(self):
        if self._alpha_map is None:
            self._alpha_map = texture.get_default_texture_2d()
        return self._alpha_map
    @alpha_map.setter
    def alpha_map(self, value):
        self._alpha_map = texture.as_texture_2d(value)


@dataclass
class Scene:
    objects: Dict[str, Mesh]
    instances: List[InstanceInfo]
    camera: Camera = Camera(screen_size=(720, 1024))

def teapot_spiral():
    import matplotlib.pyplot as plt

    meshes = {
        'teapot': load_object('teapot'),
        'plane': Mesh.plane(2.0)
    }
    instanes = [
        InstanceInfo(object='plane',
                     ModelMatrix=transform.scale(80),
                     kd=[0.48856, 0.238193, 0.187987],
                     ks=[0.1, 0.1, 0.1],
                     shininess=500.0)
    ]
    # Place teapots in an upward spiral.
    num_pots = 10
    angles = np.linspace(0, np.radians(720), num=10)
    zs = np.linspace(0, 10, num=10)
    radius = 8

    colors = plt.get_cmap('tab20')(np.linspace(0, 1, num=num_pots))
    colors = colors[:, :3].astype(np.float32)

    for angle, y, color in zip(angles, zs, colors):
        x = radius * np.cos(angle)
        z = radius * np.sin(angle)
        M = transform.translate([x, y, z])
        instanes.append(InstanceInfo(object='teapot',
                                     ModelMatrix=M,
                                     kd=color))

    scene = Scene(meshes, instanes)
    scene.camera.look_at([0, 5, 0], [0, 30, 20])
    return scene    
