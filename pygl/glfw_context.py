import numpy as np
import OpenGL.GL as GL
import glfw
import logging

import imgui
from imgui.integrations.glfw import GlfwRenderer

from .base import Context

class GLFWContext(Context):
    glfw_windows = 0

    def __init__(self, shape, name="Window", visible=True):
        super(GLFWContext, self).__init__()
        self.__title = name
        assert glfw.init(), "Could not initialize GLFW"
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        if not visible:
            glfw.window_hint(glfw.VISIBLE, glfw.FALSE)        
        height, width = shape
        self._context = glfw.create_window(width, height, name, None, None)
        assert self._context, 'Could not create context/window'
        GLFWContext.glfw_windows += 1

        glfw.make_context_current(self._context)
        imgui.create_context()
        self._imgui_impl = GlfwRenderer(self._context)
        
        self.clear_color = np.array([0.1, 0.1, 0.1, 1.0], dtype=np.float32)

    def __del__(self):
        super().__del__()
        logging.debug("GLFWContext.__del__")
        glfw.destroy_window(self._context)
        GLFWContext.glfw_windows -= 1

        if GLFWContext.glfw_windows == 0:
            logging.debug("GLFWContext: termiate glfw")
            glfw.terminate()

    def _make_current(self):
        glfw.make_context_current(self._context)

    def start_frame(self):
        if self.should_close:
            return False
        glfw.poll_events()
        imgui.new_frame()
        self._imgui_impl.process_inputs()
        
        cur_size = self.size[::-1]
        GL.glViewport(0, 0, *cur_size)
        GL.glEnable(GL.GL_DEPTH_TEST)

        GL.glClearColor(*self.clear_color)
        GL.glClear(GL.GL_DEPTH_BUFFER_BIT | GL.GL_COLOR_BUFFER_BIT)
        return not self.should_close

    def end_frame(self):
        imgui.render()
        self._imgui_impl.render(imgui.get_draw_data())
        imgui.end_frame()
        glfw.swap_buffers(self._context)

    @property
    def title(self):
        return self.__title
    @title.setter
    def title(self, title):
        if self._context:
            glfw.set_window_title(self._context, title)
            self.__title = title

    @property
    def size(self):
        if self._context:
            width, height = glfw.get_window_size(self._context)
            return (height, width)
        return None

    def set_size(self, height, width):
        if self._context:
            glfw.set_window_size(self._context, width, height)

    def set_resize_callback(self, callback):
        if self._context:
            glfw.set_framebuffer_size_callback(self._context, callback)

    @property
    def should_close(self):
        if not self.valid:
            return True
        glfw.make_context_current(self._context)
        return glfw.window_should_close(self._context)

    @property
    def valid(self):
        return self._context != None
