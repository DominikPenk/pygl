import logging
from tkinter import Frame

import numpy as np
import OpenGL.GL as gl

from .base import Context, GLObject
from .texture import Texture2D, tfilter, tformat, ttype


class FrameBuffer(GLObject):
    def __init__(self, shape):
        super(FrameBuffer, self).__init__(gl.glGenFramebuffers, gl.glDeleteFramebuffers)

        self.attachments = {}
        self.__color_buffers = set()
        self.__size = shape
        self.__default_depth_rbo = 0
        self._are_draw_buffers_attached = True
        self._is_read_buffer_attached = True

        self.__default_readbuffer = gl.GLenum(gl.glGetIntegerv(gl.GL_READ_BUFFER))
        
    def _check_status(self):
        status = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
        if status == gl.GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
            raise RuntimeError("Framebuffer incomplete: Attachment is NOT complete.")
        elif status == gl.GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
            raise RuntimeError("Framebuffer incomplete: No image is attached to FBO.")
        elif status == gl.GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER:
            raise RuntimeError("Framebuffer incomplete: Draw buffer is NOT complete.")
        elif status == gl.GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER:
            raise RuntimeError("Framebuffer incomplete: Read buffer is NOT complete.")
        elif status == gl.GL_FRAMEBUFFER_UNSUPPORTED:
            raise RuntimeError("Unsupported by FBO implementation.")
        elif status == gl.GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE:
            raise RuntimeError("Framebuffer incomplete: Multisample settings invalid.")
        elif status == gl.GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS:
            raise RuntimeError("Framebuffer incomplete: Layer and renderbuffer attachment mismatch.")


    def _attach_buffers(self):
        # Do we need to attach a default renderbuffer (since no attachments were specified)?
        if not self.has_depth_texture and self.__default_depth_rbo == 0:
            self.__default_depth_rbo = gl.glGenRenderbuffers(1)
            gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self.__default_depth_rbo)
            gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_DEPTH_COMPONENT24, self.width, self.height)
            gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, 
                                         gl.GL_RENDERBUFFER, self.__default_depth_rbo)
            gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, 0)

        if self.__color_buffers:
            color_buffers = list(self.__color_buffers)
            gl.glDrawBuffers(len(self.__color_buffers), color_buffers)
        self._are_draw_buffers_attached = True

    def free(self):
        super().free()
        if self.__default_depth_rbo != 0:
            gl.glDeleteRenderbuffers(1, np.array(self.__default_depth_rbo))
        
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
        if slot not in [gl.GL_DEPTH_ATTACHMENT, gl.GL_DEPTH_STENCIL_ATTACHMENT]:
            self.__color_buffers.add(slot)
        else:
            # Check if we we had a default depth renderbuffer and delete it
            if self.__default_depth_rbo != 0:
                gl.glDeleteRenderbuffers(1, np.array(self.__default_depth_rbo))
                self.__default_depth_rbo = 0

        texture_target = kwargs.get('texture_target', texture._type)
        
        # TODO: This is ugly we should probably only use glFramebufferTexture2D if 
        # The target is a cube map side
        with self:
            if texture_target == gl.GL_TEXTURE_CUBE_MAP:
                gl.glFramebufferTexture(gl.GL_FRAMEBUFFER, slot, texture.id, mip_level)
            else:
                texture.bind()
                gl.glFramebufferTexture2D(
                    gl.GL_FRAMEBUFFER, 
                    slot, 
                    texture_target,
                    texture.id, 
                    mip_level)
                texture.unbind()      
            self._attach_buffers()
            self._check_status()
        return texture
        
    def detach_draw_buffers(self):
        if not self._are_draw_buffers_attached:
            logging.warning("Draw buffer is already detached")

        with self:
            self._are_draw_buffers_attached = False
            gl.glDrawBuffer(gl.GL_NONE)

    def attach_draw_buffers(self):
        if self._are_draw_buffers_attached:
            logging.warning("Draw buffers are already attached")
        with self:
            self._attach_buffers()

    def detach_read_buffer(self):
        if not self._is_read_buffer_attached:
            logging.warning("Read buffer is already detached")

        with self:
            self._is_read_buffer_attached = False
            gl.glReadBuffer(gl.GL_NONE)
    
    def attach_read_buffer(self, target:gl.GLenum=None):
        if target is None:
            target = self.__default_readbuffer
        with self:
            self._is_read_buffer_attached = True
            gl.glReadBuffer(target)

    def resize(self, shape):
        if shape[1] != self.width or shape[0] != self.height:
            self.__size = shape
            for _, tex in self.attachments.items():
                tex.resize(self.width, self.height)
            # Resize the default color and depth buffers
            if self.__default_depth_rbo != 0:
                gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self.__default_depth_rbo)
                gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_DEPTH_COMPONENT24, self.width, self.height)
            gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, 0)
            

    def bind(self):
        should_be_bound = Context.current().try_push_fbo(self)
        if should_be_bound:
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.id)
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

    @property
    def is_readbuffer_attached(self):
        return self._is_read_buffer_attached
    
    @property
    def are_draw_buffers_attached(self):
        return self._are_draw_buffers_attached

    @property
    def is_complete(self):
        with self:
            return gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) == gl.GL_FRAMEBUFFER_COMPLETE