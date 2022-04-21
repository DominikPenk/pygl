import logging as log
import os
import time

import numpy as np
import OpenGL.GL as gl

from .base import GLObject, context_cached
from .buffers import ShaderStorageBuffer
from .texture import Texture2D, TextureBase

SHADER_TYPE_MAP = {
    # Short names
    'vs': gl.GL_VERTEX_SHADER,
    'tcs': gl.GL_TESS_CONTROL_SHADER,
    'tes': gl.GL_TESS_EVALUATION_SHADER,
    'gs': gl.GL_GEOMETRY_SHADER,
    'fs': gl.GL_FRAGMENT_SHADER,
    'cs': gl.GL_COMPUTE_SHADER,
    'compute': gl.GL_COMPUTE_SHADER,

    # Long names
    'vertex': gl.GL_VERTEX_SHADER,
    'tess_control': gl.GL_TESS_CONTROL_SHADER,
    'tess_eval': gl.GL_TESS_EVALUATION_SHADER,
    'geometry': gl.GL_GEOMETRY_SHADER,
    'fragment': gl.GL_FRAGMENT_SHADER,
    'compute': gl.GL_COMPUTE_SHADER  
}

def type_from_path(path):
    ext = path[path.rfind('.')+1:]
    return SHADER_TYPE_MAP.get(ext, None)

def type_from_key(key):
    return SHADER_TYPE_MAP.get(key, None)

class Shader(GLObject):
    def __init__(self, *sources, **kwargs):
        super(Shader, self).__init__()
        assert not (len(sources) != 0 and len(kwargs) != 0), "You may only create shaders from files or sources"
        if len(sources) > 0: 
            self.__shaders = [
                (s, type_from_path(s), None) for s in sources if type_from_path(s) is not None
            ]
            self.__from_files = True
        else:
            self.__shaders = [
                (source, type_from_key(key), None) for key, source in kwargs.items() if type_from_key(key) is not None
            ]
            self.__from_files = False
    
    def free(self):
        log.debug("Shader.free")
        if self.id != 0:
            gl.glDeleteProgram(self.id)
            self._id = 0

    def _compile(self):
        shader_ids = []
        
        now = time.time()

        for (path_or_src, shader_type, _) in self.__shaders:
            shader_ids.append(Shader.__create_shader__(path_or_src, shader_type))

        if self.id != 0:
            gl.glDeleteProgram(self.id)
        self._id = gl.glCreateProgram()
        for shader_id in shader_ids:
            gl.glAttachShader(self.id, shader_id)

        gl.glLinkProgram(self.id)
        if not gl.glGetProgramiv(self.id, gl.GL_LINK_STATUS):
            log.error(gl.glGetProgramInfoLog(self.id))
            raise RuntimeError('Shader linking failed')
        else:
            log.debug('Shader linked.')

        for shader_id in shader_ids:
            gl.glDeleteShader(shader_id)

        self.__shaders = [(s, t, now) for (s, t, _) in self.__shaders]

        # Store uniform types
        n = gl.glGetProgramiv(self.id, gl.GL_ACTIVE_UNIFORMS, None)
        self._uniform_types = { k.decode(): v for k, _, v in  [gl.glGetActiveUniform(self.id, i) for i in range(n)]}

    @staticmethod
    def __read_file__(path):
        with open(path, 'r') as f:
            data = f.read()
        return data

    @staticmethod
    def __create_shader__(path_or_src, shader_type):
        from_file = os.path.isfile(path_or_src)
        if from_file:
            shader_code = Shader.__read_file__(path_or_src)
        else:
            shader_code = path_or_src
        shader = gl.glCreateShader(shader_type)
        gl.glShaderSource(shader, shader_code)
        gl.glCompileShader(shader)
        if not gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS):
            msg = f"[{path_or_src}]: Shader failed to compile!" if from_file else "Static shader failed to compile!"
            log.error(msg)
            log.error(f"{gl.glGetShaderInfoLog(shader).decode()}")
            raise RuntimeError(msg)
        elif from_file:
            log.debug(f"[{path_or_src}]: Shader compiled!")
        return shader

    @property
    def up_to_date(self):
        if self.id == 0:
            return False
        elif not self.__from_files:
            return True
        else:
            shaders_updated = [ lu is None or lu < os.path.getmtime(s) for (s, _, lu) in self.__shaders ]
            return not True in shaders_updated

    @property
    def is_compute(self):
        return self.__shaders[0][1] == gl.GL_COMPUTE_SHADER

    @property
    def workgroup_size(self):
        if not self.is_compute:
            return (None, None, None)
        else:
            import ctypes

            from OpenGL.GL import GL_COMPUTE_WORK_GROUP_SIZE
            from OpenGL.raw.GL.VERSION.GL_2_0 import glGetProgramiv
            if not self.up_to_date:
                self._compile()
            sizes = np.ones([3], dtype=np.int32)
            glGetProgramiv(self.id, GL_COMPUTE_WORK_GROUP_SIZE, sizes)
            return sizes
        
    def use(self, **kwargs):
        if not self.up_to_date:
            self._compile()
        gl.glUseProgram(self.id)
        self.set_uniforms(**kwargs)

    def dispatch(self, x, y=1, z=1, **kwargs):
        assert self.is_compute, "Shader is not a compute shader"
        self.use(**kwargs)
        gl.glDispatchCompute(x, y, z)
        gl.glMemoryBarrier(gl.GL_SHADER_STORAGE_BARRIER_BIT)

    def dispatch_on_array(self, array, **kwargs):
        assert self.is_compute, "Shader is not a compute shader"
        if isinstance(array, Texture2D):
            shape = array.shape[:3]
        else:
            array = np.asanyarray(array)
            shape = array.shape[:3]
        shape = np.array(shape + (1,) * (3 - len(shape)))
        workgroup_shape = self.workgroup_size
        ny, nx, nz = np.ceil(np.array(shape) / np.array(workgroup_shape)).astype(np.int32)
        self.dispatch(nx, ny, nz, **kwargs)


    def set_float(self, name, value):
        gl.glUniform1f(gl.glGetUniformLocation(self.id, name), value)

    def set_int(self, name, value):
        gl.glUniform1i(gl.glGetUniformLocation(self.id, name), value)

    def set_vector(self, name, vector):
        if vector.ndim == 1:        
            if vector.size == 1:
                gl.glUniform1f(gl.glGetUniformLocation(self.id, name), *vector)
            elif vector.size == 2:
                gl.glUniform2f(gl.glGetUniformLocation(self.id, name), *vector)
            elif vector.size == 3:
                gl.glUniform3f(gl.glGetUniformLocation(self.id, name), *vector)
            elif vector.size == 4:
                gl.glUniform4f(gl.glGetUniformLocation(self.id, name), *vector)
            else:
                raise RuntimeError("Invalid value")
        elif vector.ndim == 2:
            n = len(vector)
            dim = vector.shape[-1]
            if dim == 1:
                gl.glUniform1fv(gl.glGetUniformLocation(self.id, name), n, vector)
            elif dim == 2:
                gl.glUniform2fv(gl.glGetUniformLocation(self.id, name), n, vector)
            elif dim == 3:
                gl.glUniform3fv(gl.glGetUniformLocation(self.id, name), n, vector)
            elif dim == 4:
                gl.glUniform4fv(gl.glGetUniformLocation(self.id, name), n, vector)
            else:
                raise RuntimeError("Invalid value")

    def set_matrix(self, name, value):
        assert isinstance(value, np.ndarray), 'Matrix must be a numpy array'
        assert value.ndim == 2, 'Matrix must be a 2D array'
        if value.shape == (2, 2):
            gl.glUniformMatrix2fv(gl.glGetUniformLocation(self.id, name), 1, gl.GL_TRUE, value)
        elif value.shape == (3, 3):
            gl.glUniformMatrix3fv(gl.glGetUniformLocation(self.id, name), 1, gl.GL_TRUE, value)
        elif value.shape == (4, 4):
            gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.id, name), 1, gl.GL_TRUE, value)
        else:
            raise RuntimeError(f"Matrix shape {value.shape} not supported")

    def set_matrices(self, name, matrices):
        assert isinstance(matrices, np.ndarray), 'Matrices must be a numpy array'
        assert matrices.ndim == 3, 'Matrices must be a 3D array'

        matrix_shape = matrices.shape[1:]
        n = matrices.shape[0]


        if matrix_shape == (2, 2):
            bind_fn = gl.glUniformMatrix2fv
        elif matrix_shape == (3, 3):
            bind_fn = gl.glUniformMatrix3fv
        elif matrix_shape == (4, 4):
            bind_fn = gl.glUniformMatrix4fv
        else:
            raise RuntimeError(f"Matrix shape {matrices.shape} not supported")

        for i, matrix in enumerate(matrices):
            bind_fn(gl.glGetUniformLocation(self.id, f"{name}[{i}]"), 1, gl.GL_TRUE, matrix)

    def set_sampler(self, name, texture, slot):
        texture.bind(slot)
        gl.glUniform1i(gl.glGetUniformLocation(self.id, name), slot)

    def set_uniforms(self, **kwargs):
        """Set multiple uniforms for shader
        The types are guessed from data type of arguments
        """
        next_texture_slot = 0
        next_free_ssbo_slot = 0
        for name, value in kwargs.items():
            if isinstance(value, (list, tuple)):
                value = np.asanyarray(value, dtype=np.float32)

            try:
                if isinstance(value, float):
                    gl.glUniform1f(gl.glGetUniformLocation(self.id, name), value)
                elif isinstance(value, (int, bool)):
                    gl.glUniform1i(gl.glGetUniformLocation(self.id, name), value)
                elif isinstance(value, np.ndarray):
                    if value.dtype != np.float32:
                        print(f"Set uniform {name}: Unsupported format {value.dtype}. Going to convert")
                        value = value.astype(np.float32)
                    if value.ndim == 1:
                        self.set_vector(name, value)
                    elif value.ndim == 2:
                        self.set_matrix(name, value)
                    elif value.ndim == 3:
                        self.set_matrices(name, value)
                    else:
                        raise RuntimeError(f"Unsupported array shape {value.shape}")
                elif isinstance(value, TextureBase):
                    uniform_type = self._uniform_types.get(name, None)
                    if uniform_type in [gl.GL_SAMPLER_2D, gl.GL_SAMPLER_CUBE, None]:
                        value.bind(next_texture_slot)
                        gl.glUniform1i(gl.glGetUniformLocation(self.id, name), next_texture_slot)
                    else:
                        gl.glBindImageTexture(next_texture_slot, value.id, 0, False, 0, gl.GL_WRITE_ONLY, value.sized_format)
                    next_texture_slot += 1  
                elif isinstance(value, ShaderStorageBuffer):
                    block_index = gl.glGetProgramResourceIndex(self.id, gl.GL_SHADER_STORAGE_BLOCK, name)
                    value.bind(next_free_ssbo_slot)
                    gl.glShaderStorageBlockBinding(self.id, block_index, next_free_ssbo_slot)
                    next_free_ssbo_slot += 1
            except gl.GLError as e:
                msg = f"Failed to set uniform {name}\n{e.description.decode()}"
                print(msg)
                log.error(msg)
                raise e

def get_bling_phong_shader(lighting_mode='spherical_harmonics'):
    """Return a Phong shader with Bling lighting model
    """
    assert lighting_mode in ['spherical_harmonics']
    from .shaders.phong_shader import phong_fs, phong_vs

    # TODO: Different lighting models
    return Shader(vertex=phong_vs, fragment=phong_fs)

@context_cached
def get_pbr_shader():
    from .shaders.pbr_shader import pbr_fs, pbr_vs
    return Shader(vertex=pbr_vs, fragment=pbr_fs)

@context_cached
def get_flat_shader():
    vertex_shader = """#version 430
    layout (location=0) in vec3 pos;
    uniform mat4 mvp;
    void main() { gl_Position = mvp * vec4(pos, 1.0); }
    """
    fragment_shader = """#version 430
    out vec4 FragColor;
    uniform vec4 color;
    void main() { FragColor = color; }
    """
    return Shader(vertex=vertex_shader, fragment=fragment_shader)

@context_cached
def get_flat_with_vc():
    vertex_shader = """#version 430
    layout (location=0) in vec3 pos;
    layout (location=2) in vec3 vc;
    layout (location=3) in vec3 fc;

    out vec4 color;
    uniform mat4 mvp;
    uniform int use_face_color;
    void main() { 
        gl_Position = mvp * vec4(pos, 1.0);
        color = use_face_color == 1 ? vec4(fc, 1.0) : vec4(vc, 1.0); 
    }
    """
    fragment_shader = """#version 430
    out vec4 FragColor;
    in vec4 color;
    void main() { FragColor = color; }
    """
    return Shader(vertex=vertex_shader, fragment=fragment_shader)

def get_normal_phong_shader():
    from .shaders.normal_shader import normal_fs, normal_vs
    return Shader(vertex=normal_vs, fragment=normal_fs)

def get_flat_normal_shader():
    vertex_shader = """#version 430
    layout (location=0) in vec3 pos;
    layout (location=2) in vec3 aNormal;

    out vec3 normal;
    uniform mat4 mvp;
    uniform mat4 model;

    void main() { 
        gl_Position = mvp * vec4(pos, 1.0);
        normal = normalize(mat3(model) * aNormal);   
    }
    """
    fragment_shader = """#version 430
    out vec4 FragColor;
    in vec3 normal;
    void main() { FragColor = vec4(normal, 1.0); }
    """
    return Shader(vertex=vertex_shader, fragment=fragment_shader)

def get_flat_nocs_shader():
    vertex_shader = """#version 430
    layout (location=0) in vec3 pos;

    out vec3 coord;
    uniform mat4 mvp;
    uniform vec3 pmin;
    uniform vec3 pmax;
    void main() { 
        gl_Position = mvp * vec4(pos, 1.0);
        coord = (pos - pmin) / (pmax - pmin);
    }
    """
    fragment_shader = """#version 430
    out vec4 FragColor;
    in vec3 coord;
    void main() { FragColor = vec4(coord, 1.0); }
    """
    return Shader(vertex=vertex_shader, fragment=fragment_shader)

def get_shadow_map_shader():
    shadow_map_vs = """#version 330 core
    layout (location = 0) in vec3 pos;
    uniform mat4 mvp;
    void main() {
        gl_Position = mvp * vec4(pos, 1.0);
    }
    """
    shadow_map_fs = """#version 330 core
    void main() {}"""

    return Shader(vertex=shadow_map_vs, fragment=shadow_map_fs)

__all__ = [
    'Shader',
    'get_pbr_shader'
    'get_flat_shader',
    'get_flat_with_vc'
]
