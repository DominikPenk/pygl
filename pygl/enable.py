from OpenGL import GL

def _to_gl_const(name, throw=True):
    try:
        return getattr(GL, name.upper())
    except AttributeError as e:
        if throw:
            raise e
        else:
            return None

class EnableGuard(object):
    def __init__(self, *args, **kwargs):
        self.enables = { name : True for name in args}
        self.enables.update({ _to_gl_const(name) : value
                             for name, value in kwargs.items() })

        self.previous = {}

    def __enter__(self):
        for name, value in self.enables.items():
            self.previous[name] = GL.glIsEnabled(name)
            if value:
                GL.glEnable(name)
            else:
                GL.glDisable(name)

            if value in [GL.GL_FRONT, GL.GL_BACK, GL.GL_FRONT_AND_BACK]:
                self.previous[name] = GL.glGetIntegerv(GL.GL_CULL_FACE_MODE)
                GL.glCullFace(value)

    def __exit__(self, exception_type, exception_value, exception_traceback):
        for name, value in self.previous.items():
            if value:
                GL.glEnable(name)
            else:
                GL.glDisable(name)

            if value in [GL.GL_FRONT, GL.GL_BACK, GL.GL_FRONT_AND_BACK]:
                self.previous[name] = GL.glGetIntegerv(GL.GL_CULL_FACE_MODE)
                GL.glCullFace(value)

def enable(*args, **kwargs):
    return EnableGuard(*args, **kwargs)

def enables(*args, **kwargs):
    guard = EnableGuard(*args, **kwargs)
    def decorator(fn):
        def wrapped(*args, **kwargs):
            with guard:
                result = fn(*args, **kwargs)
            return result
        return wrapped
    return decorator
