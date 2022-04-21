import abc
import logging
import weakref

import numpy as np
import OpenGL.GL as gl

from . import glstate


class GLObject(object):
    def __init__(self, constructor=None, destructor = None):
        self._id = constructor(1) if constructor is not None else 0

        self._destructor = destructor
        ctx = Context.current()
        assert ctx is not None, "No current context"
        ctx._register_created_object(self)

    def free(self):
        if self._id != 0 and self._destructor is not None:
            self._destructor(1, np.array(self._id))
            self._id = 0
    
    def __del__(self):
        try:
            self.free()
        except Exception:
            pass 

    @property
    def id(self):
        return self._id

class Context(abc.ABC, glstate._GLState):
    context_stack = []
    __refs__ = []

    @staticmethod
    def current():
        return Context.context_stack[-1] if len(Context.context_stack) > 0 else None 

    @classmethod
    def get_instances(cls):
        cls.__refs__ = list(filter(lambda ref: ref is not None, cls.__refs__))
        return cls.__refs__

    def __init__(self):
        """Initialization should always be called at the end of the subclass constructor"""
        self.__created_objects = []
        self.fbo_stack = []
        self._dummy_vao = 0
        
        Context.__refs__.append(weakref.ref(self))

        self.set_active()
        glstate._GLState.__init__(self, True)

    def __del__(self):
        logging.debug("Context.__del__")
        with self as ctx:
            for obj in ctx.__created_objects:
                if obj is not None:
                    logging.debug(f"Context.__del__: free {obj.__class__.__name__}:{obj().id}")
                    obj().free()
            self.__created_objects = []
        # remove every other reference in context stack
        Context.context_stack = [ctx for ctx in Context.context_stack if ctx != self]
        if self._dummy_vao != 0:
            gl.glDeleteVertexArrays(1, np.array(self._dummy_vao))

    @abc.abstractmethod
    def _make_current(self):
        pass

    def try_push_fbo(self, fbo)->bool:
        """Try to push a new fbo onto the stack.
        If it is already on top. Nothing will be pushed.
        
        Returns if the fbo was pushed onto the stack."""
        is_new_fbo = len(self.fbo_stack) == 0 or self.fbo_stack[-1].id != fbo.id
        self.fbo_stack.append(weakref.proxy(fbo))
        return is_new_fbo

    def pop_fbo(self, fbo)->None:
        """Pops fbo from the stack. If fbo and the top of bof stack differ, this 
        raises an exception"""
        # TODO: Implement custom exception
        if len(self.fbo_stack) == 0:
            raise RuntimeError("Tried to pop from empty FBO stack")
        if self.fbo_stack[-1].id != fbo.id:
            msg = f"FBO stack was corrupted. Tried to pop {fbo.id} but top of stack is {self.fbo_stack[-1].id}"
            raise RuntimeError(msg)
        
        self.fbo_stack.pop()

        if len(self.fbo_stack) > 0:
            new_fbo = self.fbo_stack[-1]
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, new_fbo.id)
            gl.glViewport(0, 0, new_fbo.width, new_fbo.height)
        else:
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

    def set_active(self):
        if len(Context.context_stack) == 0 or Context.context_stack[-1] != self:
            self._make_current()
        Context.context_stack.append(self)
    
    def dismiss(self):
        assert Context.context_stack[-1] == self, "Context stack was corrupted"
        Context.context_stack = Context.context_stack[:-1]
        if len(Context.context_stack) > 0 and Context.context_stack[-1] != self:
            Context.context_stack[-1].set_active()
        logging.debug(f"New context stack size {len(Context.context_stack)}")

    def __enter__(self):
        logging.debug("Context.__enter__")
        self.set_active()
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        logging.debug("Context.__exit__")
        self.dismiss()

    def _register_created_object(self, obj):
        assert isinstance(obj, GLObject), "Tried to register instance of a non GLObject"
        self.__created_objects.append(weakref.ref(obj))

    @property 
    def dummy_vao(self):
        if self._dummy_vao == 0:
            self._dummy_vao = gl.glGenVertexArrays(1)
        return self._dummy_vao


def context_cached(func, cache=None):
    if cache is None:
        setattr(func, '_context_cache', {})
        cache = func._context_cache
    def wrapped(*args, **kwargs):
        use_cache = kwargs.pop('use_cache', True)
        if use_cache:
            ctx = Context.current()
            if not ctx in cache:
                cache[ctx] = func(*args, **kwargs)
            return cache[ctx]
        else:
            return func(*args, **kwargs)
    return wrapped
