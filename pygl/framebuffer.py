import logging
import numpy as np
import OpenGL.GL as gl

from pygl.base import GLObject, Context
from pygl.texture import Texture2D, tfilter, tformat, ttype

class FrameBuffer(GLObject):
    def __init__(self, shape):
        super(FrameBuffer, self).__init__(gl.glGenFramebuffers, gl.glDeleteFramebuffers)

        self.attachments = {}
        self.__color_buffers = []
        self.__size = shape
        self.__rbo = 0

        with self:
            if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE:
                raise RuntimeError("Failed to initialize FBO")
    
    def free(self):
        super().free()
        if self.__rbo != 0:
            gl.glDeleteRenderbuffers(1, np.array(self.__rbo))
        
    def attach_texture(self, slot, texture = None, mip_level=0, **kwargs):
        if not (gl.GL_COLOR_ATTACHMENT0 <= slot <= gl.GL_COLOR_ATTACHMENT0 + gl.GL_MAX_COLOR_ATTACHMENTS):
            slot = gl.GL_COLOR_ATTACHMENT0 + slot
        if texture is None:
            # TODO: check that mip_level is 0
            if 'tformat' in kwargs:
                tf = kwargs['tformat']
            elif slot == gl.GL_DEPTH_ATTACHMENT:
                tf = tformat.depth
            elif slot == gl.GL_DEPTH_STENCIL_ATTACHMENT:
                tf = tformat.depth_stencil
            else:
                tf = tformat.rgba 

            if 'dtype' in kwargs:
                tp=ttype.from_dtype(kwargs['dtype'])
                assert 'tp' not in kwargs, "Both `dtype` and `tp` are specified. Only use one."
            elif 'tp' in kwargs:
                tp = kwargs['tp']
            elif slot == gl.GL_DEPTH_ATTACHMENT:
                tp = ttype.float32
            elif slot == gl.GL_DEPTH_STENCIL:
                tp = ttype.uint24_8
            else:
                tp = ttype.uint8
            
            texture = Texture2D(self.size, 
                                tfilter=tfilter.nearest,
                                tformat=tf,
                                tp=tp)
        else:
            pass
            # print(f"Using texture: {texture.__class__.__name__}.{texture.id} [ttype: {texture.ttype.name}]")
        self.attachments[slot] = texture
        if slot not in [gl.GL_DEPTH_ATTACHMENT, gl.GL_DEPTH_STENCIL_ATTACHMENT] and slot not in self.__color_buffers:
            self.__color_buffers.append(slot)
        self.bind()
        texture.bind()
        textarget = kwargs.get("texture_target", gl.GL_TEXTURE_2D)
        gl.glFramebufferTexture2D(
            gl.GL_FRAMEBUFFER, 
            slot, 
            textarget,
            texture.id, 
            mip_level)      
        if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("Failed to create FBO")
        texture.unbind()
        self.unbind()
        return texture
        
    def resize(self, shape):
        if shape[1] != self.width or shape[0] != self.height:
            self.__size = shape
            for _, tex in self.attachments.items():
                tex.resize(self.width, self.height)
            
            if self.__rbo != 0:
                gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self.__rbo)
                gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_DEPTH24_STENCIL8, self.width, self.height)
                gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, 0)            

    def bind(self):
        should_be_bound = Context.current().try_push_fbo(self)
        if should_be_bound:
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.id)

            if not self.has_depth_texture and self.__rbo == 0:
                self.__rbo = gl.glGenRenderbuffers(1)
                gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self.__rbo)
                gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_DEPTH_COMPONENT24, self.width, self.height)
                gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_RENDERBUFFER, self.__rbo)
            if self.__color_buffers:
                gl.glDrawBuffers(len(self.__color_buffers), self.__color_buffers)

        gl.glViewport(0, 0, self.size[1], self.size[0])

    def unbind(self):
        ctx = Context.current()
        ctx.pop_fbo(self)
        
    def clear_color_attachment(self, slot, color):
        gl.glClearBufferfv(gl.GL_COLOR, slot, np.array(color).astype(np.float32))

    def clear_depth_buffer(self):
        gl.glClear(gl.GL_DEPTH_BUFFER_BIT)

    def __enter__(self):
        self.bind() 
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        self.unbind()

    @property
    def width(self):
        return self.__size[1]
    
    @property
    def height(self):
        return self.__size[0]

    @property
    def size(self):
        return self.__size

    @size.setter
    def size(self, s):
        self.resize(s)

    @property
    def has_depth_texture(self):
        return gl.GL_DEPTH_ATTACHMENT in self.attachments or gl.GL_DEPTH_STENCIL_ATTACHMENT in self.attachments