import numpy as np
import OpenGL.GL as gl

from . import get_offscreen_context
from .camera import Camera
from .framebuffer import FrameBuffer


class VisualizationContext(object):
    stack = []

    def __init__(self, shape, **kwargs):
        self._ctx = get_offscreen_context(shape)
        self._ctx.set_active()

        fbo_dtype = kwargs.pop("fbo_dtype", np.uint8)
        self.fbo = FrameBuffer(self.size)
        self.tex = self.fbo.attach_texture(gl.GL_COLOR_ATTACHMENT0,
                                           dtype=fbo_dtype)

        self.depth_test  = kwargs.get('depth_test', True)
        self.culling     = kwargs.get('culling', True)
        clear_color      = kwargs.pop('clear_color', [0, 0, 0, 1])
        clear_color      = np.asanyarray(clear_color, dtype=np.float32)
        self.clear_color = kwargs.get('clear_color', clear_color) 
        self.camera      = Camera(self.size)

        self.start()

    def get_result(self):
        img = self.tex.download()
        return img

    def start(self):
        VisualizationContext.stack.append(self)
        self._ctx.set_active()
        self.fbo.bind()
        if self.depth_test:
            gl.glEnable(gl.GL_DEPTH_TEST)
        else:
            gl.glDisable(gl.GL_DEPTH_TEST)
        if self.culling:
            gl.glEnable(gl.GL_CULL_FACE)
            gl.glCullFace(gl.GL_BACK)
        # print("Clear color", self.clear_color)
        gl.glClearColor(*self.clear_color)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

    def end(self, download=True):
        if download:
            result = self.get_result()
        else:
            result = None
        self._ctx.dismiss()
        VisualizationContext.stack.pop()
        return result

    @classmethod
    def active(cls):
        if len(cls.stack) == 0:
            return None
        else:
            return cls.stack[-1]

    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, e_type, e_val, e_trace):
        self.end(download=False)

    @property
    def size(self):
        return self._ctx.size
    @size.setter
    def size(self, value):
        self._ctx.size = value
    @property
    def width(self):
        return self._ctx.size[1]
    @property
    def height(self):
        return self._ctx.size[0]

__all__ = [ 'VisualizationContext' ]
