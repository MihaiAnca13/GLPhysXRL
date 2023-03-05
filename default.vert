#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;
layout (location = 2) in vec3 aNormal;

out vec3 vertexColor;
out vec3 Normal;

uniform mat4 camMatrix;
uniform mat4 model;

void main()
{
    gl_Position = camMatrix * model * vec4(aPos, 1.0);

    vec4 normal = model * vec4(aNormal, 0.0);
    Normal = normalize(normal.xyz);
    vertexColor = aColor;
};