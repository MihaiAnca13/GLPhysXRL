#version 330 core

in vec3 vertexColor;
in vec3 Normal;

out vec4 FragColor;

// constants
vec3 lightDirection = normalize(vec3(1.0f, 1.0f, 0.0f));
vec4 lightColor = vec4(1.0f, 1.0f, 1.0f, 1.0f);
// ambient lighting
float ambient = 0.20f;

vec4 direcLight()
{
    // diffuse lighting
    vec3 normal = normalize(Normal);
    float diffuse = max(dot(normal, lightDirection), 0.0f);

    vec4 color = vec4(vertexColor.x, vertexColor.y, vertexColor.z, 1.0f);

    return (color * (diffuse + ambient)) * lightColor;
}

void main()
{
    FragColor = direcLight();
};