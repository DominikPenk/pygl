from pathlib import Path

import numpy as np
import trimesh
import trimesh.creation
import trimesh.visual


class Mesh(trimesh.Trimesh):
    @classmethod
    def from_trimesh(cls, mesh):
        if hasattr(mesh, 'visual') and isinstance(mesh.visual, trimesh.visual.color.ColorVisuals):
            face_colors   = mesh.visual.face_colors
            vertex_colors = mesh.visual.vertex_colors
        else:
            face_colors = None
            vertex_colors=None     

        my_mesh = Mesh(vertices=mesh.vertices,
                    faces=mesh.faces,
                    face_normals=mesh._cache['face_normals'],
                    vertex_normals=mesh._cache['vertex_normals'],
                    face_colors=face_colors,
                    vertex_colors=vertex_colors,
                    face_attributes=mesh.face_attributes,
                    vertex_attributes=mesh.vertex_attributes,
                    metadata=mesh.metadata,
                    process=False)
        if hasattr(mesh, 'visual'):
            my_mesh.visual = mesh.visual
        return my_mesh

    @classmethod
    def load(cls, path):
        if not isinstance(path, (str, Path)):
            raise ValueError("path must be a string or Path")
        else:
            mesh = trimesh.load(path, force='mesh')
            return cls.from_trimesh(mesh)

    @classmethod
    def cube(cls, size=2.0):
        mesh = trimesh.creation.box(extents=np.array([size, size, size]))
        return cls.from_trimesh(mesh)

    @classmethod
    def plane(cls, size=2.0):
        # vertices of the plane
        vertices = np.array([[-0.5, 0, -0.5],
                             [ 0.5, 0, -0.5],
                             [ 0.5, 0,  0.5],
                             [-0.0, 0,  0.5]], dtype=np.float64)
        vertices = size * vertices

        faces = np.array([[0, 1, 3],
                          [1, 2, 3]], dtype=np.int64)
        
        face_normals = np.array([[0, 1, 0],
                                 [0, 1, 0]], dtype=np.float32)
        
        plane = trimesh.Trimesh(vertices=vertices,
                                faces=faces,
                                face_normals=face_normals,
                                process=False)
        return cls.from_trimesh(plane)


    def __init__(self,
                 vertices=None,
                 faces=None,
                 A=None,
                 face_normals=None,
                 vertex_normals=None,
                 face_colors=None,
                 vertex_colors=None,
                 face_attributes=None,
                 vertex_attributes=None,
                 metadata=None,
                 process=True,
                 validate=False,
                 initial_cache=None,
                 visual=None,
                 **kwargs):

        assert vertices.ndim == 2 and faces.ndim == 2, "vertices and faces must be of rank 3"
        vertices = np.array(vertices)
        faces    = np.array(faces)
        self.dim = vertices.shape[-1] 
        # Zero pad if 2d
        if self._dim == 2:
            vertices = np.concatenate([
                vertices,
                np.zeros((len(vertices), 1), dtype=vertices.dtype)
            ], axis=-1)
        super(Mesh, self).__init__(vertices=vertices, faces=faces, 
                                   face_normals=face_normals, vertex_normals=vertex_normals, 
                                   face_colors=face_colors, vertex_colors=vertex_colors, 
                                   face_attributes=face_attributes, 
                                   vertex_attributes=vertex_attributes, 
                                   metadata=metadata, 
                                   process=process, 
                                   validate=validate, 
                                   initial_cache=initial_cache, 
                                   visual=visual)

    def render(self, shader=None, **kwargs):
        if not hasattr(self, '_render_data'):
            # Setup rendering state of this mesh
            from .render_data import MeshRenderData
            self._render_data = MeshRenderData(self)

        self._render_data.render(shader=shader, **kwargs)  
    
    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        self.add_attribute(key, value)

    def __setattr__(self, name: str, value):
        prefix, _,  key = name.partition("_")
        if (name in ['metadata', 'vertex_attributes', 'face_attributes', 'edge_attributes'] or 
            name in dir(self) or
            prefix not in ['v', 'vertex', 'e', 'edge', 'f', 'face', 'm', 'mesh', 'metadata']):
            super().__setattr__(name, value)
        else:
            if prefix in ['v', 'vertex']:
                assert value.shape[0] == self.n_vertices, "Number of vertices do not match the attribute" 
                self.vertex_attributes[key] = value
            elif prefix in ['f', 'face']:
                assert value.shape[0] == self.n_faces, "Number of faces do not match the attribute" 
                self.face_attributes[key] = value
            elif prefix in ['e', 'edge']:
                assert value.shape[0] == self.n_edges, "Number of faces do not match the attribute" 
                self.edge_attributes[key] = value
            elif prefix in ['m', 'mesh', 'meta']:
                self.metadata[key] = value
            else:
                super().__setattr__(name, value)

    @property
    def vertices(self):
        return self._data.get('vertices', np.empty(shape=(0, 3), dtype=np.float64))[:, :self._dim]

    @vertices.setter
    def vertices(self, values):
        if values.shape[-1] == 2:
            values = np.concatenate([values, 
                                     np.zeros([len(values), 1], dtype=values.dtype)
                                    ], axis=-1)
        self._data['vertices'] = np.asanyarray(values, order='C', dtype=np.float64)
        
        # Notify rendering if available
        if hasattr(self, '_render_data'):
            self._render_data.update_geometry()

    @property
    def n_vertices(self):
        return len(self.vertices)
    @property
    def n_edges(self):
        return len(self.edges_unique)

    @property
    def n_faces(self):
        return len(self.faces)

    @property
    def boundary_vertices(self):
        if not hasattr(self, '_boundary_verts'):
            unique_edges, counts = np.unique(self.mesh.edges_sorted, return_counts=True, axis=0)
            boundary_edges = unique_edges[counts == 1]
            self._boundary_verts = np.unique(boundary_edges)
        return self._boundary_verts
    
    @property
    def boundary_mask(self):
        mask = np.full((self.n_vertices, ), False, dtype=np.bool)
        mask[self.boundary_vertices] = True
        return mask

    @property
    def dim(self):
        return self._dim
    @dim.setter
    def dim(self, value):
        assert 2 <= value <= 3, f"Dimension must be 2 or 3, got {value}"
        self._dim = value

    @property
    def mesh_attributes(self):
        return self.metadata

    @property
    def face_colors(self):
        if hasattr(self, "visual") and isinstance(self.visual, trimesh.visual.color.ColorVisuals):
            return self.visual.face_colors.astype(np.float32) / 256.0
        else:
            return np.ones((self.n_faces, 4), dtype=np.float32)

    @property
    def vertex_colors(self):
        if hasattr(self, "visual") and isinstance(self.visual, trimesh.visual.color.ColorVisuals):
            return self.visual.vertex_colors.astype(np.float32) / 256.0
        else:
            return np.ones((self.n_vertices, 4), dtype=np.int32)

    @property
    def uvs(self):
        if (hasattr(self, "visual") and 
            isinstance(self.visual, trimesh.visual.texture.TextureVisuals) and
            hasattr(self.visual, 'uv') and 
            self.visual.uv is not None):
            return self.visual.uv
        else:
            return np.zeros((self.n_vertices, 2), dtype=np.int32)

    @property
    def pmin(self):
       return np.min(self.vertices, axis=0)
    @property
    def pmax(self):
         return np.max(self.vertices, axis=0)
