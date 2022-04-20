from collections import namedtuple
from cv2 import equalizeHist
from dataclasses import dataclass
from enum import IntEnum
from typing import Union, Tuple

import OpenGL.GL as gl

class CompareFunc(IntEnum):
    never = gl.GL_NEVER
    less   = gl.GL_LESS
    equal = gl.GL_EQUAL
    less_or_equal = gl.GL_LEQUAL
    greater = gl.GL_GREATER
    not_equal = gl.GL_NOTEQUAL
    greater_or_equal = gl.GL_GEQUAL
    always = gl.GL_ALWAYS

class CullingMode(IntEnum):
    front = gl.GL_FRONT
    back = gl.GL_BACK
    front_and_back = gl.GL_FRONT_AND_BACK
    none = 0

class BlendFactor(IntEnum):
    zero = gl.GL_ZERO
    one = gl.GL_ONE
    src_color = gl.GL_SRC_COLOR
    one_minus_src_color = gl.GL_ONE_MINUS_SRC_COLOR
    dst_color = gl.GL_DST_COLOR
    one_minus_dst_color = gl.GL_ONE_MINUS_DST_COLOR
    src_alpha = gl.GL_SRC_ALPHA
    one_minus_src_alpha = gl.GL_ONE_MINUS_SRC_ALPHA
    dst_alpha = gl.GL_DST_ALPHA
    one_minus_dst_alpha = gl.GL_ONE_MINUS_DST_ALPHA
    constant_color = gl.GL_CONSTANT_COLOR
    one_minus_constant_color = gl.GL_ONE_MINUS_CONSTANT_COLOR
    constant_alpha = gl.GL_CONSTANT_ALPHA
    one_minus_constant_alpha = gl.GL_ONE_MINUS_CONSTANT_ALPHA

class BlendEquation(IntEnum):
    add = gl.GL_FUNC_ADD
    subtract = gl.GL_FUNC_SUBTRACT
    reverse_subtract = gl.GL_FUNC_REVERSE_SUBTRACT
    min = gl.GL_MIN
    max = gl.GL_MAX

class StencilOp(IntEnum):
    keep = gl.GL_KEEP
    zero = gl.GL_ZERO
    replace = gl.GL_REPLACE
    increment = gl.GL_INCR
    decrement = gl.GL_DECR
    invert = gl.GL_INVERT
    increment_wrap = gl.GL_INCR_WRAP
    decrement_wrap = gl.GL_DECR_WRAP

@dataclass
class BlendMode:
    equation: BlendEquation      = BlendEquation.add
    factor_rgb_src:BlendFactor   = BlendFactor.one
    factor_rgb_dst:BlendFactor   = BlendFactor.zero
    factor_alpha_src:BlendFactor = BlendFactor.one
    factor_alpha_dst:BlendFactor = BlendFactor.zero

    @property
    def needs_enable(self):
        return self != BlendMode()


class _GLState(object):
    """This object handels internal state of an OpenGL context.
    You should not need to manually use this one but handle it via the
    Context class."""
    def __init__(self, apply_defaults=True):
        self._cull_face:CullingMode  = CullingMode.back
        self._depth_test:CompareFunc = CompareFunc.less
        self._write_depth:bool       = True
        self._blending:BlendMode     = BlendMode()

        # Stencil Setting
        self._stencil_func:CompareFunc   = CompareFunc.always
        self._stencil_ref:int            = 0
        self._stencil_mask:int           = 0xFF # TODO: Get the default value from OpenGL
        self._stencil_op_sfail:StencilOp = StencilOp.keep
        self._stencil_op_dfaul:StencilOp = StencilOp.keep
        self._stencil_op_pass:StencilOp  = StencilOp.keep

        if apply_defaults:
            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glEnable(gl.GL_CULL_FACE)
    
            gl.glDisable(gl.GL_BLEND)
            
            gl.glDisable(gl.GL_STENCIL_TEST)
            gl.glDepthMask(self._write_depth)
   
    @property
    def write_depth(self)->bool:
        return self._write_depth
    @write_depth.setter
    def write_depth(self, value:bool):
        if value != self._write_depth:
            gl.glDepthMask(gl.GL_TRUE if value else gl.GL_FALSE)
            self._write_depth = value    

    @property
    def is_depth_test_enabled(self)->bool:
        return self._depth_test is not CompareFunc.always

    @property
    def depth_test(self)->CompareFunc:
        return self._depth_test
    @depth_test.setter
    def depth_test(self, value:Union[CompareFunc, None]):
        if value is None:
            value = CompareFunc.always
        if value != self._depth_test:
            gl.glDepthFunc(value)
            self._depth_test = value
    
    @property
    def is_culling_enabled(self):
        return self._cull_face is not CullingMode.none
    @property
    def culling(self)->CullingMode:
        return self._cull_face
    @culling.setter
    def culling(self, value:Union[CullingMode, None]):
        if value is None:
            value = CullingMode.none
        
        if value != self._cull_face and value == CullingMode.none:
            gl.glDisable(gl.GL_CULL_FACE)
            self._cull_face = value
        elif value != self._cull_face:
            if self._cull_face == CullingMode.none:
                gl.glEnable(gl.GL_CULL_FACE)
            gl.glCullFace(value)
            self._cull_face = value
        
    def set_stencil_op(self, 
                       sfail:StencilOp=StencilOp.keep, 
                       dfail:StencilOp=StencilOp.keep, 
                       success:StencilOp=StencilOp.keep):
        if sfail != self._stencil_op_sfail or \
            dfail != self._stencil_op_dfaul or \
            success != self._stencil_op_pass:
            gl.glStencilOp(sfail, dfail, success)
            self._stencil_op_sfail = sfail
            self._stencil_op_dfaul = dfail
            self._stencil_op_pass = success

    def set_stencil_func(self, 
                         stencil_func:CompareFunc=CompareFunc.always, 
                         ref:int=0, 
                         mask:int=0xFF):
        if self._stencil_func != stencil_func or \
            self._stencil_ref != ref or \
            self._stencil_mask != mask:
            if self._stencil_func == CompareFunc.always:
                gl.glDisable(gl.GL_STENCIL_TEST)
            else:
                gl.glEnable(gl.GL_STENCIL_TEST)
            gl.glStencilFunc(stencil_func, ref, mask)
            self._stencil_func = stencil_func
            self._stencil_ref = ref
            self._stencil_mask = mask

    def set_stencil_mask(self, mask:int):
        gl.glStencilMask(mask)
    def disable_stencil_write(self):
        self.set_stencil_mask(0x00)
    def enable_stencil_write(self):
        self.set_stencil_mask(0xFF)

    @property
    def is_stencil_disabled(self):
        return self._stencil_func is CompareFunc.always

    def set_blending(self, 
                     source_factor:BlendFactor = BlendFactor.src_alpha,
                     destination_factor:BlendFactor = BlendFactor.one_minus_src_alpha,
                     equation:BlendEquation = BlendEquation.add,
                     source_factor_alpha:Union[None, BlendFactor] = None,
                     destination_factor_alpha:Union[None, BlendFactor] = None):
        """Set a new blending mode"""
        if source_factor_alpha is None:
            source_factor_alpha = source_factor
        if destination_factor_alpha is None:
            destination_factor_alpha = destination_factor

        new_blending = BlendMode(
            equation=equation,
            factor_rgb_src=source_factor,
            factor_rgb_dst=destination_factor,
            factor_alpha_src=source_factor_alpha,
            factor_alpha_dst=destination_factor_alpha)

        if new_blending != self._blending:
            new_needs_enable = new_blending.needs_enable
            old_needs_enable = self._blending.needs_enable
            if new_needs_enable and not old_needs_enable:
                gl.glEnable(gl.GL_BLEND)
            elif not new_needs_enable and old_needs_enable:
                gl.glDisable(gl.GL_BLEND)

            gl.glBlendEquation(new_blending.equation)
            gl.glBlendFuncSeparate(new_blending.factor_rgb_src, new_blending.factor_rgb_dst, 
                                   new_blending.factor_alpha_src, new_blending.factor_alpha_dst)

            self._blending = new_blending
    
    def disable_blending(self):
        """Disable blending"""
        if self._blending.needs_enable:
            gl.glDisable(gl.GL_BLEND)
            self._blending = BlendMode()

    @property
    def is_blending_enabled(self)->bool:
        return self._blending.needs_enable
    @property
    def blend_mode(self)->BlendMode:
        return self._blending
    
