//
// Created by mihai on 06/03/23.
//

#include <utility>
#include "MyObjects.h"

MyObjects::MyObjects(std::vector<float> vertices, std::vector<unsigned int> indices) {
    Initialise(std::move(vertices), std::move(indices));
}

void MyObjects::Initialise(std::vector<float> vertices, std::vector<unsigned int> indices) {
    _numIndices = (GLsizei) indices.size();

    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, (GLsizei) (vertices.size() * sizeof(float)), (float*) &vertices[0], GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, (GLsizei) (indices.size() * sizeof(unsigned int)), (unsigned int*) &indices[0], GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void *) nullptr);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void *) (3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void *) (6 * sizeof(float)));
    glEnableVertexAttribArray(2);
}

void MyObjects::Draw(unsigned int shaderProgram) const {
    glBindVertexArray(VAO);

    glm::mat4 model = glm::mat4(1.0f);
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));

    glDrawElements(GL_TRIANGLES, _numIndices, GL_UNSIGNED_INT, nullptr);
}

void MyObjects::Delete() {
    // unbind buffers and delete them
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
}