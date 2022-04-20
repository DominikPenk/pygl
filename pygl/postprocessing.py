import os
import re
from typing import Union

from numpy.typing import ArrayLike

import pygl.shader as shader_lib
import pygl.texture as texture_lib

# TODO: Support non RGBA textures

# Filter code is defined by a function called apply
# This function must have a single argument which is either a vec4 color or 
# an ivec2 pixel coordinate and return a vec4 color.
_FILTER_SHADER_SRC_COLOR = """#version 430 core
layout(local_size_x = 16, local_size_y = 16) in;

layout (binding = 0) uniform sampler2D texture_in;
layout (rgba32f, binding = 1) uniform image2D texture_out;

<<<USER_CODE>>>

void main() {
    ivec2 image_size = textureSize(texture_in, 0);
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    if (pixel.x >= 0 && 
        pixel.y >= 0 &&
        pixel.y < image_size.y && 
        pixel.x < image_size.x) {
        
        vec4 in_color  = texelFetch(texture_in, pixel, 0);
        vec4 out_color = apply(in_color); 

        imageStore(texture_out, pixel, out_color);
    }
}"""

_FILTER_SHADER_SRC_IVEC2 = """#version 430 core
layout(local_size_x = 16, local_size_y = 16) in;

layout (binding = 0) uniform sampler2D texture_in;
layout (rgba32f, binding = 1) uniform image2D texture_out;

<<<USER_CODE>>>

void main() {
    ivec2 image_size = textureSize(texture_in, 0);
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    if (pixel.x >= 0 && 
        pixel.y >= 0 &&
        pixel.y < image_size.y && 
        pixel.x < image_size.x) {
        
        vec4 out_color = apply(pixel); 

        imageStore(texture_out, pixel, out_color);
    }
}"""

class Filter(object):
    """Wrapps a compute shader to apply a postprocessing filter"""
    def __init__(self, code:str, 
                 pixel_local:bool=False,
                 **kwargs):
        self._pixel_local = pixel_local
        self.filter_args = kwargs

        self._source = code
        self._is_static = not os.path.isfile(self._source)
        self._shader = self._create_shader()

    def _create_shader(self):
        # TODO: Implement automatic filter reloading
        if not self._is_static:
            with open(self._source, 'r') as f:
                code = f.read()
        else:
            code = self._source

        regex = r"vec4 apply\((vec4|ivec2)\s*\w+\)\s*{" if self._pixel_local \
                else r"vec4 apply\((ivec2)\s*\w+\)\s*{"

        match = re.search(regex, code)
        if match and match.group(1) == 'vec4':
            shader_src = _FILTER_SHADER_SRC_COLOR.replace("<<<USER_CODE>>>", code)
        elif match and match.group(1) == "ivec2":
            shader_src = _FILTER_SHADER_SRC_IVEC2.replace("<<<USER_CODE>>>", code)
        else:
            raise RuntimeError("Could not find a valid apply method in the provided filter code.")
        return shader_lib.Shader(compute=shader_src)
            
    def apply(self, 
              texture:Union[ArrayLike, texture_lib.Texture2D],
              out_texture:Union[None, texture_lib.Texture2D]=None, 
              **kwargs)->texture_lib.Texture2D:
        """Applies the filter to a texture"""
        texture = texture_lib.as_texture_2d(texture)

        if out_texture is None and self.pixel_local:
            out_texture = texture 
        elif out_texture is None:
            out_texture = texture_lib.texture_like(texture)

        if out_texture.format != texture_lib.tformat.rgba:
            raise NotImplementedError("We currently only support filtering for RGBA textures.")

        if kwargs:
            shader_args = self.filter_args.copy()
            shader_args.update(kwargs)
        else:
            shader_args = self.filter_args

        self._shader.dispatch_on_array(texture, 
                                       texture_in=texture,
                                       texture_out=out_texture,
                                       **shader_args)
        return out_texture

    @property
    def pixel_local(self)->bool:
        return self._pixel_local

    @property
    def is_static_filter(self)->bool:
        return self._is_static
    @property
    def is_from_file(self)->bool:
        return not self._is_static

