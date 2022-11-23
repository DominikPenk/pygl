#version 430

uniform sampler2D depth_texture;
uniform sampler2D color_texture;
uniform sampler2D normal_texture;

uniform vec4 intrinsics;    // Intrinsics of the RGBD-Camera used to capture depth_map
uniform mat4 view;
uniform mat4 projection;
uniform float expansion;

out float point_size;
out vec4 point_color;
out vec4 point_normal;


vec3 unproject(float x, float y, float depth) {
    return depth * vec3((x + 0.5 - intrinsics.z) / intrinsics.x,
                        (intrinsics.w - y - 0.5) / intrinsics.y,
                        -1.0);
}

void main() {
    ivec2 size = textureSize(depth_texture, 0);
    int x = gl_VertexID % (size.x - 1);
	int y = gl_VertexID / (size.x - 1);

    const float depth = texelFetch(depth_texture, ivec2(x, y), 0).r;

    if (depth > 0) {
        point_size = depth / intrinsics.x;
        point_size = max(1.0, expansion) * point_size;

        // Here you should project the 3d point back into the color texture
        // Instead we just colorize the point according to their position in the depth image
        point_color = vec4(
            float(x) / size.x,
            float(y) / size.y,
            0.0,
            1.0
        );
        point_normal = view * texelFetch(normal_texture, ivec2(x, y), 0);
        gl_Position = view * vec4(unproject(x, y, depth), 1.0);
    }
    else 
    {
        gl_Position = vec4(-5);
    }
}