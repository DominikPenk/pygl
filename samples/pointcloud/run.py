from typing import List
import os
import sys
import json
import numpy as np
from PIL import Image

FOLDER = os.path.dirname(__file__)
# We add the root directory of this repo to the PATH so we can find pygl
sys.path.insert(0, os.path.join(FOLDER, "..", ".."))   

import pygl
import pygl.immediate as immediate
import imgui
import OpenGL.GL as gl

SHADER_DIR = os.path.join(FOLDER, "shaders")

# To compute normals we unproject points and compute the 
# normal as the crossproduct of numerical forward differences along x and y axis
def compute_normals(depth, intrinsics, threshold=0.01):
    height, width = depth.shape[:2]
    fx, fy, cx, cy = intrinsics

    u, v = np.meshgrid(np.arange(width), np.arange(height))
    u = u.astype(np.float32)
    v = v.astype(np.float32)

    # We unproject points to the OpenGL coordinate system:
    # - Y downwards (vs upward in OpenCV)
    # - -Z forward (vs Z in OpenCV)
    points = np.stack([ (u - cx + 0.5) / fx,
                    -(v - cy + 0.5) / fy,
                    -np.ones_like(u)],
                   axis=-1)                 
    points = points * np.atleast_3d(depth) 

    mask = depth > 0.0

    normals = np.zeros_like(points)
    if mask is None:
        mask = np.ones(normals.shape[:2], dtype=np.bool)
    else:
        # Update mask to ensure that the normals are computed for all points
        mask = np.logical_and.reduce([np.roll(mask,  1, axis=0),
                                      np.roll(mask, -1, axis=0),
                                      np.roll(mask,  1, axis=1),
                                      np.roll(mask, -1, axis=1),
                                      mask])

    pc = np.roll(points, -1, axis=0)[1:-1, 1:-1]
    cp = np.roll(points, -1, axis=1)[1:-1, 1:-1]
    mc = np.roll(points,  1, axis=0)[1:-1, 1:-1]
    cm = np.roll(points,  1, axis=1)[1:-1, 1:-1]


    dz = np.maximum(np.abs(pc[..., 2] - mc[..., 2]), np.abs(cp[..., 2] - cm[..., 2]))
    ns = np.cross(pc - mc, cp - cm)
    l = np.linalg.norm(ns, axis=-1)

    valid_mask = np.logical_and.reduce([mask[1:-1, 1:-1],
                                        ~np.isclose(l, 0.0),
                                        dz < threshold
                                        ])

    ns[valid_mask] /= l[valid_mask, None]
    ns[~valid_mask] = 0.0
    normals[1:-1, 1:-1] = ns
    return normals

# Shader code for normalizing accumulated color data
def bg_normalize_fs():
    return '''
    #version 330
    in vec2 texCoord;

    uniform sampler2D image;
    layout(location = 0) out vec4 FragColor;


    void main()
    {
        vec4 col = texture(image, texCoord);
        float w = max(1e-6, col.w);
        FragColor = vec4(col.xyz / col.w, 1.0);
    }
    '''

def main():
    with open(os.path.join(FOLDER, "meta.json"), 'r') as f:
        meta = json.load(f)

    depth = Image.open(os.path.join(FOLDER, "depth.png"))
    # We stored the depth as unsigned integers scaled by depth_scale
    # Convert it back to the original depth value (in meters)
    depth = np.array(depth).astype(np.float32) * meta['depth_scale'] 
    color = np.array(Image.open(os.path.join(FOLDER, "color.png")))

    # Numpy stores image shape like this: (height, width, channels)
    # The GLFWContext constructor adopts the same format (minus the channels)
    window = pygl.GLFWContext(depth.shape[:2])
    window.set_active()

    # We want to use depth and color on the GPU so we need to create textures from the numpy arrays
    depth_tex = pygl.Texture2D.from_numpy(depth)
    color_tex = pygl.Texture2D.from_numpy(color)

    # We need the normals of each unprojected pixel to align properly. Let's compute them
    normals = compute_normals(depth, meta['depth_intrinsics'])
    normal_tex = pygl.Texture2D.from_numpy(normals)

    # There are 3 steps to rendering a point cloud:
    # 1. Get an offset depth buffer (-> depth pass)
    # 2. Shade every point and accumulate multiple close patches per pixel (-> shading pass)
    # 3. Normalize the accumulated color data
    depth_pass_shader = pygl.Shader(
        os.path.join(SHADER_DIR, "point_depth.vs"),
        os.path.join(SHADER_DIR, "point.gs"),
        os.path.join(SHADER_DIR, "point_depth.fs")
    )
    shading_pass_shader = pygl.Shader(
        os.path.join(SHADER_DIR, "point_shading.vs"),
        os.path.join(SHADER_DIR, "point.gs"),
        os.path.join(SHADER_DIR, "point_shading.fs")
    )
    normalize_shader = pygl.Shader(
        vertex=immediate.fullscreen_triangle_vs(version=330),
        fragment=bg_normalize_fs()
    )

    # We will render the point cloud in a custom frame buffer and copy the normalized version 
    # into the display framebuffer later
    fbo = pygl.FrameBuffer((480, 640))
    pc_tex = fbo.attach_texture(0, dtype=np.float32)    # Make sure that the color buffer is float

    # We need a camera to view the world from
    cam = pygl.Camera((480, 640))
    cam.look_at((0, 0, 0), (0, 1, 5))

    try:
        # This only is neccessary for compatibility profiles or gl version < 3.1
        # See: https://stackoverflow.com/a/27301058
        gl.glEnable(gl.GL_POINT_SPRITE)
    except gl.GLError:
        pass
    gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)

    # The main loop of any application should look like this.
    # window.start_frame returns False if the window should be closed (e.g. the x button was clicked)
    window.set_active()

    point_size = 1.0

    while window.start_frame():

        # We can use ImGui to create a simple interface
        # This is very helpful for debugging
        imgui.begin("Textures", True)
        _, point_size = imgui.drag_float("Point Size", point_size, min_value=1, max_value=5)
        imgui.text("Color")
        imgui.image(color_tex.id, color_tex.cols // 4, color_tex.rows // 4)
        imgui.text("Depth")
        imgui.image(depth_tex.id, depth_tex.cols // 4, depth_tex.rows // 4)
        imgui.text("Normals")
        imgui.image(normal_tex.id, normal_tex.cols // 4, normal_tex.rows // 4)
        imgui.end()

        # Rendering a point cloud
        with fbo:       # Make sure the fbo is bound

            # Depth Pass
            # - Clear and write depth
            fbo.clear_depth_buffer()
            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glDepthMask(gl.GL_TRUE)
            immediate.render_points(
                depth.shape[0]*depth.shape[1],
                depth_pass_shader,
                # Shader parameters
                view=cam.V,
                projection=cam.P,
                intrinsics=meta['depth_intrinsics'],
                expansion=point_size,
                depth_texture=depth_tex,
                normal_texture=normal_tex,
                depth_offset=0.005
            )

            # Shading pass
            # - Clear and write color
            # - Do not write depth
            fbo.clear_color_attachment(0, (0, 0, 0, 0))
            gl.glDepthMask(gl.GL_FALSE)
            gl.glEnable(gl.GL_BLEND)
            gl.glBlendFunc(gl.GL_ONE, gl.GL_ONE)
            immediate.render_points(
                depth.shape[0]*depth.shape[1],
                shading_pass_shader,
                # Shader parameters
                view=cam.V,
                projection=cam.P,
                intrinsics=meta['depth_intrinsics'],
                expansion=point_size,
                depth_texture=depth_tex,
                color_texture=color_tex,
                normal_texture=normal_tex
            )
            gl.glDisable(gl.GL_BLEND)
            gl.glDepthMask(gl.GL_TRUE)

        # Normalize accumulated color buffer and store result in default fbo
        immediate.render_fullscreen_triangle(
            0, 0,
            pc_tex.cols, pc_tex.rows,
            normalize_shader,
            uvmin=[0.0, 0.0],
            uvmax=[1.0, 1.0],
            image=pc_tex
        )

        # You need to call this function to display the newly rendered frame
        window.end_frame()


if __name__ == '__main__':
    main()