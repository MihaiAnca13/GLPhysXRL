//
// Created by mihai on 17/03/23.
//
#include "Mesh.h"


Mesh::Mesh(const std::string &name, std::vector<Vertex> vertices, std::vector<unsigned int> indices, bool headless) {
    this->name = name;
    this->vertices = std::move(vertices);
    this->indices = std::move(indices);
    this->headless = headless;

    if (!headless)
        setupMesh();
}


void Mesh::setupMesh() {
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);

    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), &vertices[0], GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int),
                 &indices[0], GL_STATIC_DRAW);

    // vertex positions
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), nullptr);
    // vertex colors
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void *) offsetof(Vertex, Color));
    // vertex normals
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void *) offsetof(Vertex, Normal));
    // vertex reflectivity
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void *) offsetof(Vertex, Reflectivity));

    GLenum error = glGetError();
    while (error != GL_NO_ERROR) {
        std::cout << "OpenGL Error: " << error << std::endl;
        error = glGetError();
    }

//    glBindVertexArray(0);
}


void Mesh::Draw(unsigned int shaderProgram) const {
    // draw mesh
    glBindVertexArray(VAO);
    glm::mat4 model = glm::mat4(1.0f);
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
    glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, nullptr);
}

void Mesh::Draw(unsigned int shaderProgram, glm::vec3 ballPosition, glm::vec3 cameraPosition) {
    glm::vec3 rayOrigin = cameraPosition;
    glm::vec3 rayDirection = glm::normalize(ballPosition - cameraPosition);


    if (name.find("Ground") == std::string::npos) {
        bool intersects = false;
        for (int i = 0; i < indices.size(); i += 3) {
            glm::vec3 vertex0 = vertices[indices[i]].Position;
            glm::vec3 vertex1 = vertices[indices[i + 1]].Position;
            glm::vec3 vertex2 = vertices[indices[i + 2]].Position;

            float distance;
            if (rayIntersectsTriangle(rayOrigin, rayDirection, vertex0, vertex1, vertex2, distance)) {
                if (distance <= minDistance) {
                    intersects = true;
                    break;
                }
            }
        }

        if (intersects) {
            return;
        }
    }

    // draw mesh
    glBindVertexArray(VAO);
    glm::mat4 model = glm::mat4(1.0f);
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
    glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, nullptr);
}

void Mesh::Delete() {
    // unbind buffers and delete them
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
}

// Moller-Trumbore intersection algorithm
bool Mesh::rayIntersectsTriangle(glm::vec3 rayOrigin,
                           glm::vec3 rayDirection,
                           glm::vec3 vertex0,
                           glm::vec3 vertex1,
                           glm::vec3 vertex2,
                           float &distance) {
    const float EPSILON = 0.0000001;
    glm::vec3 edge1 = vertex1 - vertex0;
    glm::vec3 edge2 = vertex2 - vertex0;
    glm::vec3 perpendicularVector = glm::cross(rayDirection, edge2);
    float determinant = glm::dot(edge1, perpendicularVector);

    if (determinant > -EPSILON && determinant < EPSILON)
        return false;

    float inverseDeterminant = 1.0 / determinant;
    glm::vec3 distanceVector = rayOrigin - vertex0;
    float u = inverseDeterminant * (glm::dot(distanceVector,perpendicularVector));

    if (u < 0.0 || u > 1.0)
        return false;

    glm::vec3 crossProductVector = glm::cross(distanceVector,edge1);
    float v = inverseDeterminant * (glm::dot(rayDirection,crossProductVector));

    if (v < 0.0 || u + v > 1.0)
        return false;

    distance = inverseDeterminant * (glm::dot(edge2,crossProductVector));

    if (distance > EPSILON)
        return true;

    else
        return false;
}