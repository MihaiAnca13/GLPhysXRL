//
// Created by mihai on 06/03/23.
//

#include <cstdio>
#include <cmath>
#include "Table.h"


Table::Table(float size, float thickness, float* color) : MyObjects() {
    std::vector<unsigned int> indices;
    std::vector<float> vertices;

    tableGeneration(indices, vertices, size, thickness, color);

    Initialise(vertices, indices);
}

void Table::Draw(unsigned int shaderProgram, const physx::PxVec3 &position) const {
    glBindVertexArray(VAO);
    glm::mat4 model = glm::mat4(1.0f);
    model = glm::translate(model, glm::vec3(position.x, position.y, position.z));
    model = glm::scale(model, glm::vec3(0.25f, 0.25f, 0.25f));
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
    glDrawElements(GL_TRIANGLES, _numIndices, GL_UNSIGNED_INT, nullptr);
}

void Table::tableGeneration(std::vector<unsigned int> &indices, std::vector<float> &vertices, float size, float thickness, float* color) {
    if (color == nullptr) {
        std::printf("No color provided!");
        throw (std::exception());
    }

    vertices = {
            // position             // color                        // normal               // reflectivity
            -size / 2, thickness, -size / 2, color[0], color[1], color[2], 0.0f, 1.0f, 0.0f, 0.0f,
            size / 2, thickness, -size / 2, color[0], color[1], color[2], 0.0f, 1.0f, 0.0f, 0.0f,
            size / 2, thickness, size / 2, color[0], color[1], color[2], 0.0f, 1.0f, 0.0f, 0.0f,
            -size / 2, thickness, size / 2, color[0], color[1], color[2], 0.0f, 1.0f, 0.0f, 0.0f,
    };

    indices = {
            0, 1, 2,
            0, 2, 3
    };
}