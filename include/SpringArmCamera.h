//
// Created by mihai on 05/03/23.
//

#ifndef C_ML_SPRING_ARM_CAMERA_H
#define C_ML_SPRING_ARM_CAMERA_H

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include <glm/gtx/vector_angle.hpp>
#include "shaderClass.h"


class SpringArmCamera {
public:
    // Stores the main vectors of the camera
    glm::vec3 Position{};
    glm::vec3 Orientation = glm::vec3(0.0f, 0.0f, -1.0f);
    glm::vec3 Up = glm::vec3(0.0f, 1.0f, 0.0f);

    // Stores the width and height of the window
    int width;
    int height;

    // Camera constructor to set up initial values
    SpringArmCamera(int width, int height, glm::vec3 position, glm::vec3 objectPos);

    // Updates and exports the camera matrix to the Vertex Shader
    void Matrix(glm::vec3 objectPos, glm::mat3 objectRot, float FOVdeg, float nearPlane, float farPlane, Shader &shader, const char *uniform);

private:
    glm::vec3 _lastObjPos{};
};

#endif //C_ML_SPRING_ARM_CAMERA_H
