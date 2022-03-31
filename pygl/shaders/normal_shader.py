normal_vs = """
#version 430
layout(location=0) in vec3 pos;
layout(location=1) in vec2 uv;
layout(location=2) in vec3 normal;

uniform mat4 model;
uniform mat4 vp;

out vec3  world_space_normal;
out vec3  world_space_pos;
out vec2  texcoord;
out vec3  vertex_color;

void main() { 
    const vec4 world_pos = model * vec4(pos, 1.0);
    gl_Position = vp * world_pos; 
    
    mat4 normal_matrix = transpose(inverse(model));

    world_space_normal =  (normal_matrix * vec4(normal, 0.0)).xyz;
    world_space_pos     = world_pos.xyz;
    vertex_color = abs(normal);
    texcoord = uv;
}
"""
normal_fs="""
#version 430
out vec4 FragColor;

in vec3 world_space_normal;
in vec3 world_space_pos;
in vec2 texcoord;
in vec3 vertex_color;

// values of the Lambertian reflectance kernel expressed in SH
const float lambert_kernel[9] = { 3.141593, 2.094395, 2.094395, 2.094395, 
                                  0.785398, 0.785398, 0.785398, 0.785398, 0.785398 };

uniform vec3  coeffs[9];
uniform float skypower;


// Default parameters
uniform vec3 kd;
uniform sampler2D kd_map;
uniform vec3 ks;
uniform sampler2D ks_map;
uniform float shininess;
uniform float alpha;
uniform sampler2D alpha_map;

uniform float ao_strength;
uniform vec3 view_pos;

const float PI = 3.1415926535897932384626433832795;
const float N0 = sqrt(1.0 / PI) / 2.0;
const float N1 = sqrt(3.0 / PI) / 2.0;
const float N2_2 = sqrt(15.0 / PI) / 4.0;
const float N2_1 = sqrt(15.0 / PI) / 2.0;
const float N2_0 = sqrt(5.0 / PI) / 4.0;

vec3 sh(const vec3 c[9], const in vec3 normal) {
    const float x = normal.x;
    const float y = normal.y;
    const float z = normal.z;

    const float xx = x*x;
    const float yy = y*y;
    const float zz = z*z;

    const float sh_00 = N0;

    const float shs[9] = {
            N0,
            N1 * y,
            N1 * z,
            N1 * x,
            N2_1 * x * y,
            N2_1 * y * z,
            N2_0 * (2.0 * zz - xx -yy),
            N2_1 * z * x,
            N2_2 * (xx - yy)
    };
    vec3 color = shs[0] * lambert_kernel[0] * c[0] +
                 shs[1] * lambert_kernel[1] * c[1] +
                 shs[2] * lambert_kernel[2] * c[2] +
                 shs[3] * lambert_kernel[3] * c[3] +
                 shs[4] * lambert_kernel[4] * c[4] +
                 shs[5] * lambert_kernel[5] * c[5] +
                 shs[6] * lambert_kernel[6] * c[6] +
                 shs[7] * lambert_kernel[7] * c[7] +
                 shs[8] * lambert_kernel[8] * c[8];
    return color;
}

void main() {
    vec3 diffuse_color  = vertex_color * texture(kd_map, texcoord).rgb * kd;
    vec3 specular_color = texture(ks_map, texcoord).rgb * ks;
    vec3 n = normalize(world_space_normal);
    vec3 v = normalize(world_space_pos-view_pos);
    vec3 r = reflect(-v, n); 

    // ambient light
    vec3 ambient = ao_strength * diffuse_color;

    // diffuse light
    vec3 diffuse = sh(coeffs, n) * skypower * diffuse_color;

    // specular light
    float s = pow(abs(dot(r, v)), shininess);
    vec3 specular = sh(coeffs, r) * skypower * specular_color * s; 

    const float alpha_channel = texture(alpha_map, texcoord).w * alpha;

    FragColor = vec4(ambient + diffuse + specular, alpha_channel);
}
"""