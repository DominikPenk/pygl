import logging
from pygl.base import Context

from pygl.mesh import Mesh
from pygl.camera import Camera
from pygl.shader import Shader
from pygl.texture import Texture2D, ttype, tformat
from pygl.enable import enable, enables

# Try to import egl
try:
    from pygl.egl_context import EGLContext
    _has_egl_context = True
except ImportError as err:
    from pygl.glfw_context import GLFWContext
    _has_egl_context = False

def get_offscreen_context(size, force_glfw=False):
    """Create a context for offscreen rendering.

    Args:
        size (tuple): (height, width)
    """
    if _has_egl_context and not force_glfw:
        return EGLContext(size)
    else:
        if not force_glfw:
            logging.warn(
                "EGL was not found. Going to use an invisible GLFW window "
                "for offscreen rendering. This does not work if you do "
                "not have an x-server.")
        return GLFWContext(size, visible=False)

def assure_context(fn):
    """Ensures that a default context is enabled.
    This also assures that it is always the same context."""
    def wrappend(*args, **kwargs):
        if not hasattr(assure_context, '__default_context'):
            # If there is a context available, use it.
            debug_msg = "Setting default context: "
            if Context.current():
                debug_msg += "Using currently active context."
                assure_context.__default_context = Context.current()
            elif len(Context.get_instances()) > 0:
                debug_msg += "Using initially created context."
                assure_context.__default_context = Context.get_instances()[0]
            # Create an offscreen context
            else:
                debug_msg += "Using a newly cerated context."
                assure_context.__default_context = get_offscreen_context((480, 640))
            logging.debug(debug_msg)
        # Ensure that the default context is active
        with assure_context.__default_context:
            return fn(*args, **kwargs)
    return wrappend
def _set_default_context(context):
    if not issubclass(context, Context):
        raise ValueError("context must be pygl.Context subclass")
    if hasattr(assure_context, '__default_context'):
        logging.warning("Setting the default context but there is already one. "\
                        "This probably leads to unwanted behaviour.")
    assure_context.__default_context = context
assure_context.set_default_context = _set_default_context

from pygl.viz_context import VisualizationContext
from pygl.glfw_context import GLFWContext        

__all__ = [
    'get_offscreen_context',
    'VisualizationContext',
    'GLFWContext'
    'Mesh',
    'Camera',
    'Shader',
    'Texture2D',
    'ttype',
    'tformat',
    'assure_context',
    'enable',
    'enables'
]
          