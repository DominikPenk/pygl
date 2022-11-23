#version 430

layout (points) in;
layout (triangle_strip, max_vertices = 4) out;

uniform mat4 projection;

in float point_size[];
in vec4 point_color[];
in vec4 point_normal[];

out vec4 frag_color;
out vec2 splat_coord;
out vec3 position;
out vec3 normal;

mat3 get_rotation(vec3 n) {
    const float c = n.z;
    const mat3 skew = mat3(
        0, 0, -n.x,
        0, 0, -n.y,
        n.x, n.y, 0);
    return mat3(1.0) + skew + skew * skew / (1 + c);
}

void main() {
    const vec3 center = gl_in[0].gl_Position.xyz;
    const float radius = point_size[0];
    normal = point_normal[0].xyz;
    frag_color = point_color[0];

    mat3 R = get_rotation(normal);

    if (gl_in[0].gl_Position.w == 1.0) {
        position = center + R * (0.5 * vec3(-radius, -radius, 0));
        gl_Position = projection * vec4(position, 1.0);
        splat_coord = vec2(-1, -1);
        EmitVertex(); 
        position = center + R * (0.5 * vec3( radius, -radius, 0));
        gl_Position = projection * vec4(position, 1.0);
        splat_coord = vec2(1, -1);
        EmitVertex(); 
        position = center + R * (0.5 * vec3(-radius,  radius, 0));
        gl_Position = projection * vec4(position, 1.0);
        splat_coord = vec2(-1, 1);
        EmitVertex(); 
        position = center + R * (0.5 * vec3( radius,  radius, 0));
        gl_Position = projection * vec4(position, 1.0);
        splat_coord = vec2(1, 1);
        EmitVertex(); 
        EndPrimitive();
    } 
}