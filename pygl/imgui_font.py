import os
import re

import imgui

from .glfw_context import GLFWContext

# TODO: Handle DPI using glfw.get_window_content_scale()

def _standardize_font_mode(mode):
    # Convert to snake case
    # See https://stackoverflow.com/a/1176023
    return re.sub(r'(?<!^)(?=[A-Z])', '_', mode).lower()

class _FontProx(object):
    """This class proxies raw imgui_fonts and provides some
    functionality for scoping.
    You should probably never initialize it by yourself"""
    def __init__(self, font):
        self.font = font

    def push(self):
        imgui.push_font(self.font)

    def pop(self):
        imgui.pop_font()

    def __enter__(self):
        imgui.push_font(self.font)
    
    def __exit__(self, type, value, traceback):
        imgui.pop_font()

class Font(object):
    def __init__(self, 
                 default="default",
                 size=16,
                 **kwargs):
        
        imgui_fonts = imgui.get_io().fonts
        self._default = default
        self._modes = {
            _standardize_font_mode(mode): imgui_fonts.add_font_from_file_ttf(font, size)
            for mode, font in kwargs.items()
        }
        if self._default not in self._modes:
            raise ValueError(f"Default font mode '{self._default}' not found")

    def __getattr__(self, font_mode):
        if font_mode in self._modes:
            return _FontProx(self._modes[font_mode])
        else:
            raise AttributeError(
                "'{}' object has no attribute '{}'".format(
                    self.__class__.__name__, font_mode))

    def __enter__(self):
        imgui.push_font(self._modes[self._default])

    def __exit__(self, type, value, traceback):
        imgui.pop_font()

    def push(self, font_mode="default"):
        imgui.push_font(self._modes[font_mode])

    def pop(self):
        imgui.pop_font()

class FontManager(object):
    """A class managing fonts for glfwWindow"""
    def __init__(self, window:GLFWContext):
        self._window = window

        self._fonts = {}

    def add_fron_from_folder(self, folder, name, 
                       default=None,
                       use=None,
                       size=16):
        """Add a font from a folder given a name
        The pattern should be <name>[-_]<mode>.ttf"""
        font_pattern = f"{name}[-_](\w+).ttf"
        font_files = {
            _standardize_font_mode(re.search(font_pattern, f).group(1)) : os.path.join(folder, f)
            for f in os.listdir(folder)
            if re.search(font_pattern, f)
        }
        # check if a file <name>.ttf exists, if so this should be the default font
        if os.path.exists(os.path.join(folder, f"{name}.ttf")):
            font_files['default'] = os.path.join(folder, f"{name}.ttf")

        if default is None:
            if "default" in font_files:
                default = "default"
            elif "regular" in font_files:
                default = "regular"
            else:
                raise RuntimeError(f"Could not find default font for font {name}")
        
        if use:
            use = [_standardize_font_mode(f) for f in use]
            font_files = { m: font_files[m] for m in use }

        name = _standardize_font_mode(name)

        self._fonts[name] = Font(default=default, size=size, **font_files)
