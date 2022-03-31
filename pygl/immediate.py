import logging
import numpy as np
import OpenGL.GL as gl
from pygl.shader import Shader
from pygl.buffers import VertexArrayObject

def fullscreen_triangle_vs(version=430):
	code = '''#version <<version>>
	out vec2 texCoord;

	uniform vec2 uvmin;
	uniform vec2 uvmax;

	void main()
	{
		float x = -1.0 + float((gl_VertexID & 1) << 2);
		float y = -1.0 + float((gl_VertexID & 2) << 1);
		texCoord = mix(uvmin, uvmax, vec2((x+1.0)*0.5, (y+1.0)*0.5));
		gl_Position = vec4(x, y, 0, 1);
	}'''
	code = code.replace("<<version>>", str(version))
	return code

def render_fullscreen_triangle(x, y, width, height, shader, **kwargs):

	if isinstance(shader, str):
		version = kwargs.pop("version", 430)
		vs = fullscreen_triangle_vs(version)
		shader = Shader(vertex=vs,
		                fragment=shader)
		if 'uvmin' not in kwargs:
			kwargs['uvmin'] = np.array([0.0, 0.0], dtype=np.float32)
		if 'uvmax' not in kwargs:
			kwargs['uvmax'] = np.array([1.0, 1.0], dtype=np.float32)

	depth_test_enabled = gl.glIsEnabled(gl.GL_DEPTH_TEST)
	depth_write_enabled = gl.glGetBooleanv(gl.GL_DEPTH_WRITEMASK)

	shader.use(**kwargs)
	gl.glViewport(x, y, width, height)
	gl.glDisable(gl.GL_DEPTH_TEST)
	gl.glDepthMask(gl.GL_FALSE)
	VertexArrayObject.bind_dummy()
	gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)

	if depth_test_enabled:
		gl.glEnable(gl.GL_DEPTH_TEST)
	gl.glDepthMask(depth_write_enabled)

def display_texture(x, y, width, height, texture, 
                    uvmin=[0.0, 0.0], uvmax=[1.0, 1.0]):
	
	fs = '''
	#version 330
	in vec2 texCoord;

	uniform sampler2D image;
	layout(location = 0) out vec4 FragColor;


	void main()
	{
		FragColor = texture(image, texCoord);
	}
	'''
	uvmin = np.asanyarray(uvmin, dtype=np.float32).flatten()
	uvmax = np.asanyarray(uvmax, dtype=np.float32).flatten()
	assert len(uvmin) == 2 and len(uvmax) == 2, "uvmin and uvmax must have length 2."
	if not hasattr(display_texture, '__shader'):
		display_texture.__shader = Shader(vertex=fullscreen_triangle_vs(version=330), fragment=fs)
	render_fullscreen_triangle(x, y, width, height,
							  display_texture.__shader,
							  uvmin=uvmin,
							  uvmax=uvmax,
							  image=texture)

def render_points(num_points, shader, **kwargs):
	shader.use(**kwargs)
	VertexArrayObject.bind_dummy()	
	gl.glDrawArrays(gl.GL_POINTS, 0, num_points)
