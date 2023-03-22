//
// Created by mihai on 17/03/23.
//

#ifndef C_ML_MESH_H
#define C_ML_MESH_H

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include "shaderClass.h"
#include <string>
#include <utility>


struct Vertex {
    glm::vec3 Position;
    glm::vec3 Color;
    glm::vec3 Normal;
    glm::float32_t Reflectivity;
};

class Mesh {
public:
    // mesh data
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;
    std::string name;
    float minDistance = 2.9f;

    Mesh(const std::string& name, std::vector <Vertex> vertices, std::vector<unsigned int> indices);

    void Draw(unsigned int shaderProgram) const;
    void Draw(unsigned int shaderProgram, glm::vec3 ballPosition, glm::vec3 cameraPosition);

    void Delete();

private:
    //  render data
    unsigned int VAO, VBO, EBO;

    void setupMesh();

    bool rayIntersectsTriangle(glm::vec3 rayOrigin,
                               glm::vec3 rayDirection,
                               glm::vec3 vertex0,
                               glm::vec3 vertex1,
                               glm::vec3 vertex2,
                               float &distance);
};


#endif //C_ML_MESH_H
