//
// Created by mihai on 11/03/23.
//

#ifndef C_ML_SKYBOX_H
#define C_ML_SKYBOX_H

#include <stb_image.h>
#include <glad/glad.h>
#include <iostream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "Camera.h"
#include "SpringArmCamera.h"


class Skybox {
public:
    unsigned int VAO, VBO, EBO, texture, shaderID;

    explicit Skybox(unsigned int shaderID);
    void Draw(Camera camera, unsigned int width, unsigned int height) const;
    void Draw(SpringArmCamera camera, unsigned int width, unsigned int height) const;

private:
    // All the faces of the cubemap (make sure they are in this exact order)
    std::string facesCubemap[6] =
            {
                    "resources/skybox/px.png",
                    "resources/skybox/nx.png",
                    "resources/skybox/py.png",
                    "resources/skybox/ny.png",
                    "resources/skybox/pz.png",
                    "resources/skybox/nz.png"
            };

    const float skyboxVertices[24] =
            {
                    //   Coordinates
                    -1.0f, -1.0f, 1.0f,//        7--------6
                    1.0f, -1.0f, 1.0f,//        /|       /|
                    1.0f, -1.0f, -1.0f,//       4--------5 |
                    -1.0f, -1.0f, -1.0f,//    | |      | |
                    -1.0f, 1.0f, 1.0f,//   | 3------|-2
                    1.0f, 1.0f, 1.0f,//    |/       |/
                    1.0f, 1.0f, -1.0f,//    0--------1
                    -1.0f, 1.0f, -1.0f
            };

    const unsigned int skyboxIndices[36] =
            {
                    // Right
                    1, 2, 6,
                    6, 5, 1,
                    // Left
                    0, 4, 7,
                    7, 3, 0,
                    // Top
                    4, 5, 6,
                    6, 7, 4,
                    // Bottom
                    0, 3, 2,
                    2, 1, 0,
                    // Back
                    0, 1, 5,
                    5, 4, 0,
                    // Front
                    3, 7, 6,
                    6, 2, 3
            };
};

#endif //C_ML_SKYBOX_H
