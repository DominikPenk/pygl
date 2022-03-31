import os, sys
# set PYOPENGL_PLATFORM variable and reload OpenGL
# PYOPENGL_PLATFORM variable is set...
ogl_module_names = list(
    k for k in sys.modules.keys() if k.startswith('OpenGL')
)
for mod_name in ogl_module_names:
    del sys.modules[mod_name]
os.environ['PYOPENGL_PLATFORM'] = 'egl'
try:
    import OpenGL.EGL as egl
except ImportError as error:
    print("No EGL found!")
    os.environ.pop('PYOPENGL_PLATFORM')
    ogl_module_names = list(
        k for k in sys.modules.keys() if k.startswith('OpenGL')
    )
    for mod_name in ogl_module_names:
        del sys.modules[mod_name]
    raise error 

from OpenGL.EGL import (
    EGL_DEFAULT_DISPLAY,
    EGL_OPENGL_API,
    EGL_NO_CONTEXT,
    EGL_NO_DISPLAY,
    EGL_NO_SURFACE,
    EGLint,
    EGLConfig,
    eglGetDisplay,
    eglInitialize,
    eglChooseConfig,
    eglCreatePbufferSurface,
    eglBindAPI,
    eglCreateContext,
    eglMakeCurrent
)
from ctypes import pointer

from pygl.base import Context

__all__ = ['EGLContext']

def egl_convert_to_int_array(dict_attrs):
    # convert to EGL_NONE terminated list
    attrs = sum(([ k, v] for k, v in dict_attrs.items()), []) + [ egl.EGL_NONE ]
    # convert to ctypes array
    return (egl.EGLint * len(attrs))(*attrs)

class EGLContext(Context):
    def __init__(self, size):
        super(EGLContext, self).__init__()
        self.height, self.width = size

        # Init EGL
        self._display = eglGetDisplay(EGL_DEFAULT_DISPLAY)
        if self._display == EGL_NO_DISPLAY:
            raise RuntimeError("Failed to initialize display")
        major, minor = EGLint(), EGLint()
        if not eglInitialize(self._display, pointer(minor), pointer(major)):
            raise RuntimeError("Failed to initialize EGL")
        

        # Create config
        egl_cfg_attribs = egl_convert_to_int_array({
            egl.EGL_RED_SIZE:           8,
            egl.EGL_GREEN_SIZE:         8,
            egl.EGL_BLUE_SIZE:          8,
            egl.EGL_ALPHA_SIZE:         8,
            egl.EGL_DEPTH_SIZE:         egl.EGL_DONT_CARE,
            egl.EGL_STENCIL_SIZE:       egl.EGL_DONT_CARE,
            egl.EGL_RENDERABLE_TYPE:    egl.EGL_OPENGL_BIT,
            egl.EGL_SURFACE_TYPE:       egl.EGL_PBUFFER_BIT
        })

        egl_config = EGLConfig()
        num_configs = EGLint()
        if not eglChooseConfig(self._display, egl_cfg_attribs, pointer(egl_config), 1, pointer(num_configs)):
            raise RuntimeError("Could not choose EGL config")
        if num_configs.value == 0:
            raise RuntimeError("Config values is 0")

        # Create the context
        surface_attribs = egl_convert_to_int_array({
            egl.EGL_WIDTH: self.width,
            egl.EGL_HEIGHT: self.height
        })
        self._surface = eglCreatePbufferSurface(
            self._display, egl_config, surface_attribs)
        if self._surface == EGL_NO_SURFACE:
            raise RuntimeError("Failed to create PbufferSurface")
        if not eglBindAPI(EGL_OPENGL_API):
            raise RuntimeError("Could not initialize EGL_API")
        self._context = eglCreateContext(self._display, egl_config, EGL_NO_CONTEXT, None)
        if self._context == EGL_NO_CONTEXT:
            raise RuntimeError("Context is EGL_NOCONTEXT")

    def _make_current(self):
        if not eglMakeCurrent(self._display, self._surface, self._surface, self._context):
            raise RuntimeError("Could not activate the egl context")
