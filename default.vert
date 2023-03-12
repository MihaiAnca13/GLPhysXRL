#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;
layout (location = 2) in vec3 aNormal;
layout (location = 3) in float aShouldReflect;

out vec3 vertexColor;
out vec3 Normal;
out vec3 crntPos;
out float shouldReflect;
// Outputs the fragment position of the light
out vec4 fragPosLight;

uniform mat4 camMatrix;
uniform mat4 model;
uniform mat4 lightProjection;

void main()
{
    crntPos = vec3(model * vec4(aPos, 1.0));
    gl_Position = camMatrix * vec4(crntPos, 1.0);

    // Calculates the position of the light fragment for the fragment shader
    fragPosLight = lightProjection * vec4(crntPos, 1.0f);

    vec4 normal = model * vec4(aNormal, 0.0);
    Normal = normalize(normal.xyz);
    vertexColor = aColor;

    shouldReflect = aShouldReflect;
};