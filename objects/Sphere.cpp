//
// Created by mihai on 06/03/23.
//

#include <cstdio>
#include <cmath>
#include "Sphere.h"


Sphere::Sphere(int numSlices, int numStacks, float radius, float *colors) : MyObjects() {
    std::vector<unsigned int> indices;
    std::vector<float> vertices;

    sphereGeneration(indices, vertices, numSlices, numStacks, radius, colors);

    Initialise(vertices, indices);
}

void Sphere::Draw(unsigned int shaderProgram, const physx::PxVec3 &position, const physx::PxQuat &rotation) const {
    glBindVertexArray(VAO);
    glm::mat4 model = glm::mat4(1.0f);
    model = glm::translate(model, glm::vec3(position.x, position.y, position.z));
    glm::mat4 ballRotationMatrix = glm::mat4_cast(glm::quat(rotation.w, rotation.x, rotation.y, rotation.z));
    model = model * ballRotationMatrix;
    model = glm::scale(model, glm::vec3(0.25f, 0.25f, 0.25f));
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
    glDrawElements(GL_TRIANGLES, _numIndices, GL_UNSIGNED_INT, nullptr);
}

void Sphere::sphereGeneration(std::vector<unsigned int> &indices, std::vector<float> &vertices, int numSlices, int numStacks, float radius, const float *color) {
    if (color == nullptr) {
        std::printf("No color provided!");
        throw (std::exception());
    }

    bool shouldReflect = false;

    for (int stack = 0; stack <= numStacks; ++stack) {
        float phi = stack * M_PI / numStacks;
        float sinPhi = sin(phi);
        float cosPhi = cos(phi);
        for (int slice = 0; slice <= numSlices; ++slice) {
            float theta = slice * 2 * M_PI / numSlices;
            float sinTheta = sin(theta);
            float cosTheta = cos(theta);

            // Calculate position and normal
            vertices.push_back(radius * sinPhi * cosTheta);
            vertices.push_back(radius * sinPhi * sinTheta);
            vertices.push_back(radius * cosPhi);

            // Add color to vertices
//            vertices.push_back(color[0]);
//            vertices.push_back(color[1]);
//            vertices.push_back(color[2]);

            if ((slice % 3 != 0) && (stack > numStacks / 4 && stack < 3 * numStacks / 4)) {
                vertices.push_back(1.0f);
                vertices.push_back(1.0f);
                vertices.push_back(1.0f);
                shouldReflect = false;
            }
            else {
                vertices.push_back(color[0]);
                vertices.push_back(color[1]);
                vertices.push_back(color[2]);
                shouldReflect = true;
            }

            // Calculate normal
            float x = sinPhi * cosTheta;
            float y = sinPhi * sinTheta;
            float z = cosPhi;
            glm::vec3 normal(x, y, z);
            normal = glm::normalize(normal);

            // Add normal to vertices
            vertices.push_back(normal.x);
            vertices.push_back(normal.y);
            vertices.push_back(normal.z);

            if (shouldReflect) {
                vertices.push_back(1.0f); // should reflect
            }
            else {
                vertices.push_back(0.0f); // shouldn't reflect
            }

            // Add indices
            if (stack != numStacks && slice != numSlices) {
                int nextStack = stack + 1;
                int nextSlice = slice + 1;
                indices.push_back(stack * (numSlices + 1) + slice);
                indices.push_back(nextStack * (numSlices + 1) + slice);
                indices.push_back(nextStack * (numSlices + 1) + nextSlice);
                indices.push_back(stack * (numSlices + 1) + slice);
                indices.push_back(nextStack * (numSlices + 1) + nextSlice);
                indices.push_back(stack * (numSlices + 1) + nextSlice);
            }
        }
    }
}