#version 330 core

in vec3 vertexColor;
in vec3 Normal;
in vec3 crntPos;

out vec4 FragColor;

uniform vec3 camPos;

// constants
const vec3 lightDirection = normalize(vec3(1.0f, 1.0f, 0.0f));
const vec4 lightColor = vec4(1.0f, 1.0f, 1.0f, 1.0f);
// ambient lighting
const float ambient = 0.20f;
const float specularLight = 0.30f;

vec4 direcLight()
{
    // diffuse lighting
    vec3 normal = normalize(Normal);
    float diffuse = max(dot(normal, lightDirection), 0.0f);

    vec4 color = vec4(vertexColor.x, vertexColor.y, vertexColor.z, 1.0f);

    // specular lighting
    vec3 viewDirection = normalize(camPos - crntPos);
    vec3 reflectionDirection = reflect(-lightDirection, normal);
    float specAmount = pow(max(dot(viewDirection, reflectionDirection), 0.0f), 16);
    float specular = specAmount * specularLight;


    return (color * (diffuse + ambient + specular)) * lightColor;
}

void main()
{
    FragColor = direcLight();
};