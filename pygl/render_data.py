import numpy as np
import OpenGL.GL as gl
import ctypes

from .base import Context
from . import buffers
from .shader import get_flat_shader

class MeshRenderData(object):
    wire_shader = None

    def __init__(self, mesh):
        self.mesh = mesh
        self._face_colors   = np.ones((self.mesh.n_faces, 3), dtype=np.float32)
        self._vaos = {}
        self._buffers = {}

    def _get_packed_data(self):
        assert self.mesh is not None, "MeshRenderData has no mesh"
        assert self.mesh.uvs is not None, "MeshRenderData has no UVs"
        vertices      = self.mesh.vertices[self.mesh.faces]
        texcoords     = self.mesh.uvs[self.mesh.faces]
        normals       = self.mesh.vertex_normals[self.mesh.faces]
        vertex_colors = self.mesh.vertex_colors[:, :3][self.mesh.faces]
        face_colors   = np.repeat(self.mesh.face_colors[:, None, :3], 3, axis=1)

        vertices      = np.reshape(vertices,      (-1, 3))
        texcoords     = np.reshape(texcoords,     (-1, 2))
        normals       = np.reshape(normals,       (-1, 3))
        vertex_colors = np.reshape(vertex_colors, (-1, 3))
        face_colors   = np.reshape(face_colors,   (-1, 3))

        ids = np.arange(len(vertices))
        compact_data = np.concatenate((vertices, 
                                       texcoords, 
                                       normals, 
                                       vertex_colors,
                                       face_colors),
                                      axis=-1).astype(np.float32)
        return compact_data, ids.astype(np.uint32)

    def render(self, shader, **kwargs):
        # check if we have a vao in the current contex
        ctx = Context.current()
        if not ctx in self._vaos:
            # Create the vao for this context
            compact_data, indices = self._get_packed_data() 

            vbo = buffers.create_vbo(compact_data)
            ebo = buffers.create_index_buffer(indices)
            vao = buffers.VertexArrayObject()
            vao.setVertexAttributes(vbo, compact_data.shape[-1] * compact_data.itemsize, [
                (0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0) ,                          # Vertex Position
                (1, 2, gl.GL_FLOAT, gl.GL_FALSE, 3  * compact_data.itemsize),  # UV Coordinates
                (2, 3, gl.GL_FLOAT, gl.GL_FALSE, 5  * compact_data.itemsize),  # Vertex Normal
                (3, 3, gl.GL_FLOAT, gl.GL_FALSE, 8  * compact_data.itemsize),  # Vertex Colors
                (4, 3, gl.GL_FLOAT, gl.GL_FALSE, 11 * compact_data.itemsize)   # Face Colors
                ])
            vao.setIndexBuffer(ebo)
            self._buffers[ctx] = (vbo, ebo)
            self._vaos[ctx] = (vao, indices.size)

        vao, num_indices = self._vaos[ctx]
        vao.bind()
        shader.use(**kwargs)
        gl.glDrawElements(gl.GL_TRIANGLES, num_indices, gl.GL_UNSIGNED_INT, ctypes.c_void_p(0))
        vao.unbind()

    def render_wire(self, mvp, color, offset=-1.0):

        gl.glEnable(gl.GL_POLYGON_OFFSET_LINE)    
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        gl.glPolygonOffset(-1.0, 1.0)
        self.render(
            get_flat_shader(),
            mvp=mvp,
            color=color)
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        gl.glDisable(gl.GL_POLYGON_OFFSET_LINE)

    def update_geometry(self):
        self._vaos = {}

        compact_data = self._get_packed_data()[0]
        for vbo, _ in self._vaos.values():
            vbo.update(compact_data)
