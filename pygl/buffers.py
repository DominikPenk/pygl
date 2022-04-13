import numpy as np
import OpenGL.GL as gl

import ctypes
from .base import GLObject, Context

class BufferObject(GLObject):

    def __init__(self, data, target):
        super(BufferObject, self).__init__(gl.glGenBuffers, gl.glDeleteBuffers)
        self.__target = target
        self._size = 0
        if data is not None:
            self.resize(data.nbytes)
            self.update(data)

    def __enter__(self):
        gl.glBindBuffer(self.__target, self.id)

    def __exit__(self, type, value, traceback):
        gl.glBindBuffer(self.__target, 0)

    def bind(self, index=-1):
        gl.glBindBuffer(self.__target, self.id)
        if index >= 0:
            gl.glBindBufferBase(self.__target, index, self.id)

    def unbind(self):
        gl.glBindBuffer(self.__target, 0)

    def update(self, data, offset=0, nbytes=None):
        nbytes = data.nbytes if nbytes is None else nbytes
        self.bind()
        self._size = data.size
        gl.glBufferSubData(self.__target, offset, data.nbytes, data)
        self.unbind()

    def resize(self, nbytes):
        self.bind()
        gl.glBufferData(self.__target, nbytes, None, gl.GL_STATIC_DRAW) 
        self.unbind()

    @property
    def size(self):
        return self._size

    @property
    def target(self):
        return self.__target
    
def create_vbo(data=None):
    return BufferObject(data, gl.GL_ARRAY_BUFFER)

def create_index_buffer(data=None):
    return BufferObject(data, gl.GL_ELEMENT_ARRAY_BUFFER)

def create_ssbo(data=None):
    return BufferObject(data, gl.GL_SHADER_STORAGE_BUFFER)

class VertexArrayObject(GLObject):
    def __init__(self):
        super(VertexArrayObject, self).__init__(gl.glGenVertexArrays, gl.glDeleteVertexArrays)

    def __enter__(self):
        gl.glBindVertexArray(self.id)
        return self

    def __exit__(self, type, value, traceback):
        gl.glBindVertexArray(0)

    def bind(self):
        gl.glBindVertexArray(self.id)

    def unbind(Self):
        gl.glBindVertexArray(0)

    def setIndexBuffer(self, ebo):
        if isinstance(ebo, BufferObject):
            assert ebo.target == gl.GL_ELEMENT_ARRAY_BUFFER
            self.bind()
            ebo.bind()
            self.unbind()
            ebo.unbind
        else:
            raise('Invalid EBO type')

    def setVertexAttributes(self, vbo, stride, attribs):
        if not isinstance(vbo, BufferObject):
            raise('Ivalid VBO type')

        self.bind()
        vbo.bind()
        for (idx, dim, attr_type, normalized, rel_offset) in attribs:
            gl.glEnableVertexAttribArray(idx)
            vertexAttrib = gl.glVertexAttribIPointer if attr_type in [gl.GL_INT, gl.GL_UNSIGNED_INT] else gl.glVertexAttribPointer
            vertexAttrib(idx, dim, attr_type, normalized, stride, ctypes.c_void_p(rel_offset))
            gl.glEnableVertexAttribArray(idx)
        self.unbind()
        vbo.unbind()

    @classmethod
    def bind_dummy(cls):
        ctx = Context.current()
        assert ctx is not None, "No current context"
        gl.glBindVertexArray(ctx.dummy_vao)

class ShaderStorageBuffer(BufferObject):
    def __init__(self, data=None):
        super(ShaderStorageBuffer, self).__init__(data, gl.GL_SHADER_STORAGE_BUFFER)

    def update(self, data, offset=0, nbytes=None):
        self._dtype = data.dtype
        super().update(data, offset, nbytes)

    def download(self):
        import ctypes
        self.bind()
        ptr = gl.glMapBuffer(self.target, gl.GL_READ_ONLY)
        # Copy data into numpy array
        data = np.empty(1, self._dtype)
        ctypes.memmove(data.ctypes.data, ptr, data.nbytes)
        gl.glUnmapBuffer(self.target)
        self.unbind()
        return data