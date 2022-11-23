#version 430

in vec2 splat_coord;

void main() {
    if (dot(splat_coord, splat_coord) > 1.0) {
        discard;
    }
}