import abc
import ctypes
import logging
import os
from enum import Enum, IntEnum
from typing import Union

import imageio
import numpy as np
import OpenGL.GL as gl
from numpy.typing import ArrayLike

from .base import GLObject, context_cached
from .enable import enables


class twrap(IntEnum):
    clamp = gl.GL_CLAMP_TO_EDGE
    border = gl.GL_CLAMP_TO_BORDER
    repeat = gl.GL_REPEAT
    mclamp = gl.GL_MIRROR_CLAMP_TO_EDGE
    mrepeat = gl.GL_MIRRORED_REPEAT

    @classmethod
    def get(cls, value):
        if isinstance(value, str):
            return cls[value.lower()]
        elif isinstance(value, cls):
            return value
        else:
            raise ValueError("Invalid wrap value: {}".format(value))

class tfilter(IntEnum):
    linear = gl.GL_LINEAR
    nearest = gl.GL_NEAREST
    linear_mip_nearest = gl.GL_LINEAR_MIPMAP_NEAREST
    linear_mip_linear = gl.GL_LINEAR_MIPMAP_LINEAR
    nearest_mip_linear = gl.GL_NEAREST_MIPMAP_LINEAR
    none = 0

class tformat(IntEnum):
    red = gl.GL_RED
    rg = gl.GL_RG
    rgb = gl.GL_RGB
    bgr = gl.GL_BGR
    rgba = gl.GL_RGBA
    bgra = gl.GL_BGRA
    depth_stencil = gl.GL_DEPTH_STENCIL
    depth = gl.GL_DEPTH_COMPONENT
    default = 0

    @staticmethod
    def from_array(a : np.ndarray):
        channels = a.shape[2]
        if channels == 1:
            return tformat.red
        elif channels == 2:
            return tformat.rg
        elif channels == 3:
            return tformat.rgb
        elif channels == 4:
            return tformat.rgba
        else:
            raise ValueError("Invalid input array, last dimension must be in [1, 2, 3, 4]")

class ttype(IntEnum):
    float32 = gl.GL_FLOAT
    float16 = gl.GL_HALF_FLOAT
    uint8 = gl.GL_UNSIGNED_BYTE,
    uint24_8 = gl.GL_UNSIGNED_INT_24_8

    @classmethod
    def from_dtype(cls, dtype : np.dtype):
        if dtype == np.uint8:
            return cls.uint8
        elif dtype == np.float32:
            return cls.float32
        elif dtype == np.float16:
            return cls.float16
        else:
            return None

    @classmethod
    def from_array(cls, a : np.ndarray):
        return cls.from_dtype(a.dtype)

    @classmethod
    def to_numpy(cls, t):
        if t == ttype.float32:
            return np.float32
        elif t == ttype.uint8:
            return np.uint8
        elif t == ttype.float16:
            return np.float16
        else:
            return None

    @classmethod
    def get(cls, t):
        if isinstance(t, ttype):
            return t
        elif isinstance(t, np.dtype):
            return cls.from_dtype(t)
        elif isinstance(t, type):
            return cls.from_dtype(np.dtype(t))
        else:
            print("Cannot convert type", type(t), t.dtype)
            return None

def get_sized_format(fmt, tp):
    assert isinstance(fmt, tformat) and isinstance(tp, ttype)
    mapping = {
        ttype.float32: {
            tformat.rgb: gl.GL_RGB32F,
            tformat.bgr: gl.GL_RGB32F,
            tformat.red: gl.GL_R32F,
            tformat.rg: gl.GL_RG32F,
            tformat.rgba: gl.GL_RGBA32F,
            tformat.bgra: gl.GL_RGBA32F,
            tformat.depth: gl.GL_DEPTH_COMPONENT32F
        },
        ttype.float16: {
            tformat.rgb: gl.GL_RGB16F,
            tformat.bgr: gl.GL_RGB16F,
            tformat.red: gl.GL_R16F,
            tformat.rg: gl.GL_RG16F,
            tformat.rgba: gl.GL_RGBA16F,
            tformat.bgra: gl.GL_RGBA16F,
            tformat.depth: gl.GL_DEPTH_COMPONENT16
        },
        ttype.uint8: {
            tformat.red: gl.GL_R8,
            tformat.rg: gl.GL_RG8,
            tformat.rgb: gl.GL_RGB8,
            tformat.bgr: gl.GL_RGB8,
            tformat.rgba: gl.GL_RGBA8,
            tformat.bgra: gl.GL_RGBA8,
        },
        ttype.uint24_8: {
            tformat.depth: gl.GL_DEPTH24_STENCIL8
        }
    }
    return mapping[tp][fmt]

def get_channels(fmt):
    mapping = {
        tformat.red: 1,
        tformat.rg: 2,
        tformat.rgb: 3,
        tformat.bgr: 3,
        tformat.rgba: 4,
        tformat.bgra: 4,
        tformat.depth_stencil: 1,
        tformat.depth: 1
    }
    return mapping[fmt]

class TextureBase(GLObject, abc.ABC):
    def __init__(self, size, tex_type, fmt, tp, flt, wrap):
        super(TextureBase, self).__init__(gl.glGenTextures, gl.glDeleteTextures)
        self._size = size
        self._ttype = ttype.get(tp)
        self._tformat = fmt
        self._filter = flt
        if not isinstance(wrap, (tuple, list)):
            wrap = [wrap]
        wrap = [twrap.get(w) for w in wrap]
        self._wrap = tuple(wrap) + tuple(None for _ in range(3 - len(wrap)))
        self._type = tex_type

        # Create the texture
        resize_args = (size[1], size[0]) + size[2:]
        self.resize(*resize_args)
        self.set_wrapping(*self._wrap)
        if not isinstance(flt, (tuple, list)):
            flt = [flt]
        self.set_filter(*flt)

    def __str__(self):
        return f"<{self.__class__.__name__}: shape={self.size}, format={self.dtype}, id={self.id}>"

    def __repr__(self):
        return f"<{self.__class__.__name__}: shape={self.size}, format={self.dtype}, id={self.id}>"

    # Setters
    def set_wrapping(self, 
                     s:twrap=None, 
                     t:twrap=None,
                     r:twrap=None):
        gl.glBindTexture(self._type, self.id)
        s0, t0, r0 = self._wrap
        if s:
            gl.glTexParameteri(self._type, gl.GL_TEXTURE_WRAP_S, s)
            s0 = s
        if t:
            gl.glTexParameteri(self._type, gl.GL_TEXTURE_WRAP_T, t)
            t0 = t
        if r:
            gl.glTexParameteri(self._type, gl.GL_TEXTURE_WRAP_R, r)
            r0 = r
        self._wrap = (s0, t0, r0)
        gl.glBindTexture(self._type, 0)
        
    def set_filter(self, min_filter : tfilter, mag_filter : tfilter = None):
        if mag_filter is None:
            mag_filter = min_filter
        gl.glBindTexture(self._type, self.id)
        gl.glTexParameteri(self._type, gl.GL_TEXTURE_MIN_FILTER, min_filter)
        gl.glTexParameteri(self._type, gl.GL_TEXTURE_MAG_FILTER, mag_filter)
        gl.glBindTexture(self._type, 0)
        self._filter = (min_filter, mag_filter)
    
    @abc.abstractmethod
    def resize(self, cols, rows, depth=None):
        pass

    @property
    def sized_format(self):
        return get_sized_format(self._tformat, self._ttype)

    @property
    def format(self):
        return self._tformat

    @property
    def ttype(self):
        return self._ttype

    @property
    def filter(self):
        return self._filter

    @property
    def size(self):
        return self._size
    
    @property
    def channels(self):
        return get_channels(self._tformat)

    @property
    def cols(self):
        return self._size[1]

    @property
    def rows(self):
        return self._size[0] if len(self.size) > 1 else None

    @property
    def shape(self):
        return self._size

    @property
    def depth(self):
        return self._size[2] if len(self.size) > 2 else None

    @property
    def wrapping(self):
        return self._wrap

    @property
    def dtype(self):
        return ttype.to_numpy(self._ttype)

    def bind(self, slot=-1):
        if slot >= 0:
            gl.glActiveTexture(gl.GL_TEXTURE0 + slot)
        gl.glBindTexture(self._type, self.id)

    def unbind(self):
        gl.glBindTexture(self._type, 0)

    def generate_mipmaps(self):
        gl.glBindTexture(self._type, self._id)
        gl.glGenerateMipmap(self._type)


class Texture2D(TextureBase):
    @classmethod
    def load(cls, path, tfilter=tfilter.linear, wrap=twrap.clamp, build_mipmaps=False):
        ext = os.path.splitext(path)[-1]
        if ext == '.hdr':
            imageio.plugins.freeimage.download() # TODO only once?
        data = imageio.imread(path)
        return cls.from_numpy(data, tfilter, wrap, build_mipmaps, flip=True)

    @classmethod
    def from_numpy(cls, data, tfilter=tfilter.linear, wrap=twrap.clamp, build_mipmaps=False, flip=False):
        assert 2 <= data.ndim <= 3, f"Invalid number of dimensionst must be in [2, 3] but got {data.ndim}"
        if data.ndim == 2:
            data = data[..., np.newaxis]

        tex = cls(data.shape[:2],
                  tformat=tformat.from_array(data),
                  tp=ttype.from_array(data),
                  tfilter=tfilter,
                  wrap=wrap,
                  build_mipmaps=build_mipmaps)

        tex.update(data, flip=flip)
        return tex

    def __init__(self, 
                 shape,
                 tformat = tformat.rgba,
                 wrap = (twrap.clamp, twrap.clamp),
                 tp = ttype.uint8,
                 tfilter = tfilter.linear,
                 build_mipmaps = False):

        if not isinstance(wrap, (list, tuple)):
            wrap = (wrap, wrap)
        super(Texture2D, self).__init__(size=shape,
                                        tex_type=gl.GL_TEXTURE_2D,
                                        fmt=tformat,
                                        tp=tp,
                                        wrap=wrap,
                                        flt=tfilter)
        self.with_mipmaps = build_mipmaps

    def update(self, data, fmt = tformat.default, resize=True, flip=False):
        assert isinstance(data, np.ndarray)

        if resize and (data.shape[0] != self.rows or data.shape[1] != self.cols):
            self.resize(data.shape[1], data.shape[0])
        elif data.shape[:2] != self._size:
            data = data[0:self.rows, 0:self.cols, :]
        
        self.bind()
        fmt = self.format if fmt is tformat.default else fmt

        assert data.shape[-1] == get_channels(fmt)

        if flip:
            data = np.copy(data[::-1])
        
        old_align = gl.glGetIntegerv(gl.GL_UNPACK_ALIGNMENT)
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        gl.glPixelStorei(gl.GL_UNPACK_ROW_LENGTH, self.cols)
        gl.glTexSubImage2D(self._type, 0, 0, 0, self.cols, self.rows, fmt, self.ttype, data)
        gl.glPixelStorei(gl.GL_UNPACK_ROW_LENGTH, 0)
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, old_align)
        
        if self.with_mipmaps:
            gl.glGenerateMipmap(self._type)

        self.unbind()

    def resize(self, cols, rows, depth=None):
        if depth != None:
            logging.warning("You passed depth for resizing a 2D Texture")
        self.bind()
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, self.sized_format, cols, rows, 0, self.format, self.ttype, None)
        self._size = (rows, cols)
        self.unbind()

    def download(self, flip=True):
        self.bind()
        buf = gl.glGetTexImage(gl.GL_TEXTURE_2D,
                            0,
                            self.format,
                            self.ttype)
        img = np.reshape(np.frombuffer(buf, dtype=self.dtype),
                         (self.rows, self.cols, self.channels))
        if flip:
            img = img[::-1]
        self.unbind()
        return img

class CubeMap(TextureBase):
    from .transform import look_at
    capture_views = [
        look_at([0.0, 0.0, 0.0], [ 1.0, 0.0, 0.0], [0, -1.0, 0.0]),
        look_at([0.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0, -1.0, 0.0]),
        look_at([0.0, 0.0, 0.0], [0.0,  1.0, 0.0], [0, 0.0, 1.0]),
        look_at([0.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0, 0.0, -1.0]),
        look_at([0.0, 0.0, 0.0], [0.0, 0.0,  1.0], [0, -1.0, 0.0]),
        look_at([0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0, -1.0, 0.0])
    ]

    cubemap_vs = """
    #version 330 core
    layout (location = 0) in vec3 aPos;

    out vec3 WorldPos;

    uniform mat4 projection;
    uniform mat4 view;

    void main()
    {
        WorldPos = aPos;
        gl_Position =  projection * view * vec4(WorldPos, 1.0);
    }
    """

    convert_fs = """
    #version 330 core
    out vec4 FragColor;
    in vec3 WorldPos;

    uniform sampler2D equirectangularMap;

    const vec2 invAtan = vec2(0.1591, 0.3183);
    vec2 SampleSphericalMap(vec3 v)
    {
        vec2 uv = vec2(atan(v.z, v.x), asin(v.y));
        uv *= invAtan;
        uv += 0.5;
        return uv;
    }

    void main()
    {		
        vec2 uv = SampleSphericalMap(normalize(WorldPos));
        vec3 color = texture(equirectangularMap, uv).rgb;
        
        FragColor = vec4(color, 1.0);
    }
    """

    irradiance_convolution_fs = """
    #version 330 core
    out vec4 FragColor;
    in vec3 WorldPos;

    uniform samplerCube environmentMap;

    const float PI = 3.14159265359;

    void main()
    {		
        // The world vector acts as the normal of a tangent surface
        // from the origin, aligned to WorldPos. Given this normal, calculate all
        // incoming radiance of the environment. The result of this radiance
        // is the radiance of light coming from -Normal direction, which is what
        // we use in the PBR shader to sample irradiance.
        vec3 N = normalize(WorldPos);

        vec3 irradiance = vec3(0.0);   
        
        // tangent space calculation from origin point
        vec3 up    = vec3(0.0, 1.0, 0.0);
        vec3 right = normalize(cross(up, N));
        up         = normalize(cross(N, right));
        
        float sampleDelta = 0.025;
        float nrSamples = 0.0;
        for(float phi = 0.0; phi < 2.0 * PI; phi += sampleDelta)
        {
            for(float theta = 0.0; theta < 0.5 * PI; theta += sampleDelta)
            {
                // spherical to cartesian (in tangent space)
                vec3 tangentSample = vec3(sin(theta) * cos(phi),  sin(theta) * sin(phi), cos(theta));
                // tangent space to world
                vec3 sampleVec = tangentSample.x * right + tangentSample.y * up + tangentSample.z * N; 

                irradiance += texture(environmentMap, sampleVec).rgb * cos(theta) * sin(theta);
                nrSamples++;
            }
        }
        irradiance = PI * irradiance * (1.0 / float(nrSamples));
        
        FragColor = vec4(irradiance, 1.0);
    }

    """

    prefilter_fs = """
    #version 330 core
    out vec4 FragColor;
    in vec3 WorldPos;

    uniform samplerCube environmentMap;
    uniform float roughness;

    const float PI = 3.14159265359;
    // ----------------------------------------------------------------------------
    float DistributionGGX(vec3 N, vec3 H, float roughness)
    {
        float a = roughness*roughness;
        float a2 = a*a;
        float NdotH = max(dot(N, H), 0.0);
        float NdotH2 = NdotH*NdotH;

        float nom   = a2;
        float denom = (NdotH2 * (a2 - 1.0) + 1.0);
        denom = PI * denom * denom;

        return nom / denom;
    }
    // ----------------------------------------------------------------------------
    // http://holger.dammertz.org/stuff/notes_HammersleyOnHemisphere.html
    // efficient VanDerCorpus calculation.
    float RadicalInverse_VdC(uint bits) 
    {
        bits = (bits << 16u) | (bits >> 16u);
        bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
        bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
        bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
        bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
        return float(bits) * 2.3283064365386963e-10; // / 0x100000000
    }
    // ----------------------------------------------------------------------------
    vec2 Hammersley(uint i, uint N)
    {
        return vec2(float(i)/float(N), RadicalInverse_VdC(i));
    }
    // ----------------------------------------------------------------------------
    vec3 ImportanceSampleGGX(vec2 Xi, vec3 N, float roughness)
    {
        float a = roughness*roughness;
        
        float phi = 2.0 * PI * Xi.x;
        float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a*a - 1.0) * Xi.y));
        float sinTheta = sqrt(1.0 - cosTheta*cosTheta);
        
        // from spherical coordinates to cartesian coordinates - halfway vector
        vec3 H;
        H.x = cos(phi) * sinTheta;
        H.y = sin(phi) * sinTheta;
        H.z = cosTheta;
        
        // from tangent-space H vector to world-space sample vector
        vec3 up          = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
        vec3 tangent   = normalize(cross(up, N));
        vec3 bitangent = cross(N, tangent);
        
        vec3 sampleVec = tangent * H.x + bitangent * H.y + N * H.z;
        return normalize(sampleVec);
    }
    // ----------------------------------------------------------------------------
    void main()
    {		
        vec3 N = normalize(WorldPos);
        
        // make the simplyfying assumption that V equals R equals the normal 
        vec3 R = N;
        vec3 V = R;

        const uint SAMPLE_COUNT = 1024u;
        vec3 prefilteredColor = vec3(0.0);
        float totalWeight = 0.0;
        
        for(uint i = 0u; i < SAMPLE_COUNT; ++i)
        {
            // generates a sample vector that's biased towards the preferred alignment direction (importance sampling).
            vec2 Xi = Hammersley(i, SAMPLE_COUNT);
            vec3 H = ImportanceSampleGGX(Xi, N, roughness);
            vec3 L  = normalize(2.0 * dot(V, H) * H - V);

            float NdotL = max(dot(N, L), 0.0);
            if(NdotL > 0.0)
            {
                // sample from the environment's mip level based on roughness/pdf
                float D   = DistributionGGX(N, H, roughness);
                float NdotH = max(dot(N, H), 0.0);
                float HdotV = max(dot(H, V), 0.0);
                float pdf = D * NdotH / (4.0 * HdotV) + 0.0001; 

                float resolution = 512.0; // resolution of source cubemap (per face)
                float saTexel  = 4.0 * PI / (6.0 * resolution * resolution);
                float saSample = 1.0 / (float(SAMPLE_COUNT) * pdf + 0.0001);

                float mipLevel = roughness == 0.0 ? 0.0 : 0.5 * log2(saSample / saTexel); 
                
                prefilteredColor += textureLod(environmentMap, L, mipLevel).rgb * NdotL;
                totalWeight      += NdotL;
            }
        }

        prefilteredColor = prefilteredColor / totalWeight;

        FragColor = vec4(prefilteredColor, 1.0);
    }
    """

    brdf_fs = """
    #version 330 core
    out vec2 FragColor;
    in vec2 texCoord;

    const float PI = 3.14159265359;
    // ----------------------------------------------------------------------------
    // http://holger.dammertz.org/stuff/notes_HammersleyOnHemisphere.html
    // efficient VanDerCorpus calculation.
    float RadicalInverse_VdC(uint bits) 
    {
        bits = (bits << 16u) | (bits >> 16u);
        bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
        bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
        bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
        bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
        return float(bits) * 2.3283064365386963e-10; // / 0x100000000
    }
    // ----------------------------------------------------------------------------
    vec2 Hammersley(uint i, uint N)
    {
        return vec2(float(i)/float(N), RadicalInverse_VdC(i));
    }
    // ----------------------------------------------------------------------------
    vec3 ImportanceSampleGGX(vec2 Xi, vec3 N, float roughness)
    {
        float a = roughness*roughness;
        
        float phi = 2.0 * PI * Xi.x;
        float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a*a - 1.0) * Xi.y));
        float sinTheta = sqrt(1.0 - cosTheta*cosTheta);
        
        // from spherical coordinates to cartesian coordinates - halfway vector
        vec3 H;
        H.x = cos(phi) * sinTheta;
        H.y = sin(phi) * sinTheta;
        H.z = cosTheta;
        
        // from tangent-space H vector to world-space sample vector
        vec3 up          = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
        vec3 tangent   = normalize(cross(up, N));
        vec3 bitangent = cross(N, tangent);
        
        vec3 sampleVec = tangent * H.x + bitangent * H.y + N * H.z;
        return normalize(sampleVec);
    }
    // ----------------------------------------------------------------------------
    float GeometrySchlickGGX(float NdotV, float roughness)
    {
        // note that we use a different k for IBL
        float a = roughness;
        float k = (a * a) / 2.0;

        float nom   = NdotV;
        float denom = NdotV * (1.0 - k) + k;

        return nom / denom;
    }
    // ----------------------------------------------------------------------------
    float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
    {
        float NdotV = max(dot(N, V), 0.0);
        float NdotL = max(dot(N, L), 0.0);
        float ggx2 = GeometrySchlickGGX(NdotV, roughness);
        float ggx1 = GeometrySchlickGGX(NdotL, roughness);

        return ggx1 * ggx2;
    }
    // ----------------------------------------------------------------------------
    vec2 IntegrateBRDF(float NdotV, float roughness)
    {
        vec3 V;
        V.x = sqrt(1.0 - NdotV*NdotV);
        V.y = 0.0;
        V.z = NdotV;

        float A = 0.0;
        float B = 0.0; 

        vec3 N = vec3(0.0, 0.0, 1.0);
        
        const uint SAMPLE_COUNT = 1024u;
        for(uint i = 0u; i < SAMPLE_COUNT; ++i)
        {
            // generates a sample vector that's biased towards the
            // preferred alignment direction (importance sampling).
            vec2 Xi = Hammersley(i, SAMPLE_COUNT);
            vec3 H = ImportanceSampleGGX(Xi, N, roughness);
            vec3 L = normalize(2.0 * dot(V, H) * H - V);

            float NdotL = max(L.z, 0.0);
            float NdotH = max(H.z, 0.0);
            float VdotH = max(dot(V, H), 0.0);

            if(NdotL > 0.0)
            {
                float G = GeometrySmith(N, V, L, roughness);
                float G_Vis = (G * VdotH) / (NdotH * NdotV);
                float Fc = pow(1.0 - VdotH, 5.0);

                A += (1.0 - Fc) * G_Vis;
                B += Fc * G_Vis;
            }
        }
        A /= float(SAMPLE_COUNT);
        B /= float(SAMPLE_COUNT);
        return vec2(A, B);
    }
    // ----------------------------------------------------------------------------
    void main() 
    {
        vec2 integratedBRDF = IntegrateBRDF(texCoord.x, texCoord.y);
        FragColor = integratedBRDF;
    }
    """

    def __init__(self, size,
                 tformat:tformat = tformat.rgb,
                 wrap = (twrap.clamp, twrap.clamp, twrap.clamp),
                 tp = ttype.float32,
                 tfilter = tfilter.nearest,
                 allocate_mipmap=False):

        if not isinstance(wrap, (list, tuple)):
            wrap = (wrap, wrap, wrap)
        super(CubeMap, self).__init__(size=size,
                                     tex_type=gl.GL_TEXTURE_CUBE_MAP,
                                     fmt=tformat,
                                     tp=tp,
                                     wrap=wrap,
                                     flt=tfilter)
        if allocate_mipmap:
            self.generate_mipmaps()

    @classmethod
    def get_capture_views(cls, pos=None):
        from .transform import look_at
        if pos is None:
            pos = np.zeros((3,), dtype=np.float32)
        return [
            look_at(pos, pos+[ 1.0, 0.0, 0.0], [0, -1.0, 0.0]),
            look_at(pos, pos+[-1.0, 0.0, 0.0], [0, -1.0, 0.0]),
            look_at(pos, pos+[0.0,  1.0, 0.0], [0, 0.0, 1.0]),
            look_at(pos, pos+[0.0, -1.0, 0.0], [0, 0.0, -1.0]),
            look_at(pos, pos+[0.0, 0.0,  1.0], [0, -1.0, 0.0]),
            look_at(pos, pos+[0.0, 0.0, -1.0], [0, -1.0, 0.0])
        ]

    @classmethod
    @enables(gl_cull_face=None)
    def from_equirectengular(cls, path, size=(512, 512)):
        from . import Mesh, Shader
        from .camera import perspective
        from .framebuffer import FrameBuffer
        in_tex = Texture2D.load(path)

        fbo = FrameBuffer(size)
        envmap = cls(size,
                     tformat=tformat.rgb, 
                     tp=ttype.float16)  

        capture_projection = perspective(np.radians(90.0), 1.0, 0.1, 10.0)
        shader = Shader(vertex=cls.cubemap_vs,
                        fragment=cls.convert_fs)
        cube = Mesh.cube(size=2)
        fbo.bind()
        for i, V in enumerate(CubeMap.capture_views):
            fbo.attach_texture(gl.GL_COLOR_ATTACHMENT0, envmap, 
                               texture_target=gl.GL_TEXTURE_CUBE_MAP_POSITIVE_X + i)
            gl.glClearColor(0, 1, 0, 1)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            cube.render(shader,
                        view=V,
                        projection=capture_projection,
                        equirectangularMap=in_tex)
        fbo.unbind()
        return envmap

    @enables(gl_cull_face=None)
    def get_irradiance(self, size):
        """Computes the irradiance from the environment map"""
        from pygl import Mesh, Shader
        from .camera import perspective
        from .framebuffer import FrameBuffer

        fbo = FrameBuffer(size)
        irradiance_map = CubeMap(size=size,
                                tformat=self._tformat,
                                tp=self._ttype)
        irradiance_shader = Shader(vertex=CubeMap.cubemap_vs,
                                   fragment=CubeMap.irradiance_convolution_fs)
        irradiance_shader.use()
        capture_projection = perspective(np.radians(90.0), 1.0, 0.1, 10.0)
        cube = Mesh.cube(size=2)
        
        fbo.bind()
        for i, V in enumerate(CubeMap.capture_views):
            fbo.attach_texture(gl.GL_COLOR_ATTACHMENT0, irradiance_map, 
                               texture_target=gl.GL_TEXTURE_CUBE_MAP_POSITIVE_X + i)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            cube.render(irradiance_shader,
                        view=V,
                        projection=capture_projection,
                        environmentMap=self)
        fbo.unbind()
        return irradiance_map
        
    @enables(gl_cull_face=None)
    def get_specular_map(self, size, max_mipmap_levels=5):
        """Computes prefiltered specular maps from the environment map"""
        from . import Mesh, Shader
        from .camera import perspective
        from .framebuffer import FrameBuffer
        from .immediate import render_fullscreen_triangle
        
        specular_map = CubeMap(size=size,
                                tformat=self._tformat,
                                tp=self._ttype,
                                tfilter=(tfilter.linear_mip_linear, tfilter.linear),
                                allocate_mipmap=True)
        
        prefilter_shader = Shader(vertex=CubeMap.cubemap_vs,
                                  fragment=CubeMap.prefilter_fs)

        capture_projection = perspective(np.radians(90.0), 1.0, 0.1, 10.0)
        cube = Mesh.cube(size=2)
        
        for mip in range(max_mipmap_levels):
            mip_shape = np.asanyarray(size, dtype=np.int32) * 0.5**mip
            fbo = FrameBuffer(mip_shape.astype(np.int32))
            fbo.bind()

            roughness = mip / (max_mipmap_levels - 1.0)
            for i, V in enumerate(CubeMap.capture_views):
                fbo.attach_texture(gl.GL_COLOR_ATTACHMENT0, specular_map,
                                   mip_level=mip, 
                                   texture_target=gl.GL_TEXTURE_CUBE_MAP_POSITIVE_X + i)
                gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
                cube.render(prefilter_shader,
                            view=V,
                            projection=capture_projection,
                            roughness=roughness,
                            environmentMap=self)
            fbo.unbind()

        lut_texture = Texture2D(shape=self.size,
                                tformat=tformat.red,
                                tp=ttype.float16)
        fbo = FrameBuffer(self.size)
        fbo.attach_texture(gl.GL_COLOR_ATTACHMENT0, lut_texture)
        fbo.bind()
        render_fullscreen_triangle(0, 0, self.cols, self.rows,
                                   CubeMap.brdf_fs,
                                   version=330)
        fbo.unbind()

        return specular_map, lut_texture
            
    def resize(self, cols, rows, depth=None):
        if depth != None:
            logging.warning("You passed depth for resizing a CubeMap")
        gl.glBindTexture(self._type, self.id)
        for i in range(6):
            gl.glTexImage2D(gl.GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, self.sized_format, 
                            cols, rows, 0, self.format, self.ttype, None)
        self._size = (cols, rows)
        gl.glBindTexture(self._type, 0)

    def download(self):
        from OpenGL.raw.GL.VERSION import GL_1_1
        result = np.zeros((3*self.rows, 4*self.cols, self.channels), self.dtype)
        self.bind()
        sides = [(gl.GL_TEXTURE_CUBE_MAP_POSITIVE_X, 1, 2),
                 (gl.GL_TEXTURE_CUBE_MAP_NEGATIVE_X, 1, 0),
                 (gl.GL_TEXTURE_CUBE_MAP_POSITIVE_Y, 0, 1),
                 (gl.GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, 2, 1),
                 (gl.GL_TEXTURE_CUBE_MAP_POSITIVE_Z, 1, 1),
                 (gl.GL_TEXTURE_CUBE_MAP_NEGATIVE_Z, 1, 3)]
        buf = np.empty((self.rows, self.cols, self.channels), dtype=self.dtype)
        for side, dj, di in sides:
            GL_1_1.glGetTexImage(side,
                                 0,
                                 self.format,
                                 self.ttype,
                                 buf.ctypes.data_as(ctypes.c_void_p))
            result[dj*self.rows:(dj+1)*self.rows, 
                   di*self.cols:(di+1)*self.cols] = buf

        self.unbind()
        return result

@context_cached
def get_default_texture_2d()->Texture2D:
    return Texture2D.from_numpy(np.full((1, 1, 4), 255, dtype=np.uint8))

def as_texture_2d(source:Union[str,Texture2D,ArrayLike]):
    if isinstance(source, Texture2D):
        return source
    elif isinstance(source, str):
        return Texture2D.load(source)
    elif isinstance(source, np.ndarray):
        return Texture2D.from_numpy(source)
    else:
        raise TypeError("Cannot convert {} to Texture2D".format(type(source)))

def texture_like(source:Union[Texture2D, ArrayLike]):
    if isinstance(source, Texture2D):
        return Texture2D(source.shape, 
                         source._tformat,
                         tp=source._ttype,
                         tfilter=source._filter,
                         build_mipmaps=False)
    else:
        return Texture2D.from_numpy(np.zeros_like(np.asanyarray(source)))
