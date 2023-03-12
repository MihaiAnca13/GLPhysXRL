#version 330 core

in vec3 vertexColor;
in vec3 Normal;
in vec3 crntPos;
in float shouldReflect;
// Imports the fragment position of the light
in vec4 fragPosLight;

out vec4 FragColor;

uniform sampler2D shadowMap;
uniform vec3 camPos;
uniform vec3 lightDirection;
uniform uint specMulti;
uniform samplerCube skybox;

// constants
//const vec3 lightDirection = normalize(vec3(1.0f, 1.0f, 0.0f));
const vec4 lightColor = vec4(1.0f, 1.0f, 1.0f, 1.0f);
// ambient lighting
const float ambient = 0.20f;
const float specularLight = 0.30f;

vec4 direcLight()
{
    // diffuse lighting
    vec3 normal = normalize(Normal);
    vec3 direction = normalize(lightDirection);
    float diffuse = max(dot(normal, direction), 0.0f);

    vec4 color = vec4(vertexColor.x, vertexColor.y, vertexColor.z, 1.0f);

    // specular lighting
    vec3 viewDirection = normalize(camPos - crntPos);
    vec3 reflectionDirection = reflect(-direction, normal);
    float specAmount = pow(max(dot(viewDirection, reflectionDirection), 0.0f), specMulti);
    float specular = specAmount * specularLight;

    // Shadow value
    float shadow = 0.0f;
    // Sets lightCoords to cull space
    vec3 lightCoords = fragPosLight.xyz / fragPosLight.w;
    if(lightCoords.z <= 1.0f)
    {
        // Get from [-1, 1] range to [0, 1] range just like the shadow map
        lightCoords = (lightCoords + 1.0f) / 2.0f;
        float currentDepth = lightCoords.z;
        // Prevents shadow acne
        float bias = max(0.025f * (1.0f - dot(normal, lightDirection)), 0.0005f);

        // Smoothens out the shadows
        int sampleRadius = 2;
        vec2 pixelSize = 1.0 / textureSize(shadowMap, 0);
        for(int y = -sampleRadius; y <= sampleRadius; y++)
        {
            for(int x = -sampleRadius; x <= sampleRadius; x++)
            {
                float closestDepth = texture(shadowMap, lightCoords.xy + vec2(x, y) * pixelSize).r;
                if (currentDepth > closestDepth + bias)
                shadow += 1.0f;
            }
        }
        // Get average shadow
        shadow /= pow((sampleRadius * 2 + 1), 2);

    }

    // reflection
    vec4 reflectionColor = vec4(1.0f, 1.0f, 1.0f, 1.0f);
    if (shouldReflect > 0.0f) {
        vec3 I = normalize(crntPos - camPos);
        vec3 R = reflect(I, normal);
        reflectionColor = vec4(texture(skybox, R).rgb, 1.0f);
    }

    return (color * (diffuse * (1.0f - shadow) + ambient + specular)) * lightColor * reflectionColor;
}

void main()
{
    FragColor = direcLight();
};