#version 430

const float PI = 3.14159265359;

in vec4 frag_color;
in vec2 splat_coord;

in vec3 position;
in vec3 normal;

out vec4 FragColor;

void main() {
    float t = dot(splat_coord, splat_coord);
    if (t > 1.0) {
        discard;
    }
    vec3 Lo = frag_color.rgb;
    float alpha = exp(-t);
    FragColor = vec4(alpha*Lo, alpha);
}