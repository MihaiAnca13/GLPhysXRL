//
// Created by mihai on 05/03/23.
//
#include"SpringArmCamera.h"


SpringArmCamera::SpringArmCamera(int width, int height, glm::vec3 position, glm::vec3 objectPos) {
    SpringArmCamera::width = width;
    SpringArmCamera::height = height;
    Position = position;
    _lastObjPos = objectPos;
}


// calculate and export the camera's view and projection matrices to the vertex shader, such that the camera follows the object's position and orientation while maintaining a specified field of view and viewing frustum
void SpringArmCamera::Matrix(glm::vec3 objectPos, float FOVdeg, float nearPlane, float farPlane, Shader &shader, const char *uniform) {
    // Initializes matrices since otherwise they will be the null matrix
    glm::mat4 view = glm::mat4(1.0f);
    glm::mat4 projection = glm::mat4(1.0f);

    // update position based on objectTransform
    glm::vec3 deltaPos = objectPos - _lastObjPos;
    Position += deltaPos;
    _lastObjPos = objectPos;

    // rotate Position around Y axis using angle w.r.t. the object's position
    float x = Position.x - objectPos.x;
    float z = Position.z - objectPos.z;
    float deltaAngle = angle - _lastAngle;
    Position.x = objectPos.x + x * cos(deltaAngle) - z * sin(deltaAngle);
    Position.z = objectPos.z + x * sin(deltaAngle) + z * cos(deltaAngle);
    _lastAngle = angle;

    // Makes camera look in the right direction from the right position
    view = glm::lookAt(Position, objectPos, Up);

    // Adds perspective to the scene
    projection = glm::perspective(glm::radians(FOVdeg), (float) width / height, nearPlane, farPlane);

    // Exports the camera matrix to the Vertex Shader
    glUniformMatrix4fv(glGetUniformLocation(shader.ID, uniform), 1, GL_FALSE, glm::value_ptr(projection * view));
}

void SpringArmCamera::Inputs(GLFWwindow *window, physx::PxRigidDynamic *ball) {
    bool isChange = false;
    glm::vec3 impulseForce(0.0f);

    // Handles key inputs
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
        impulseForce.x -= force;
        isChange = true;
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
        impulseForce.x += force;
        isChange = true;
    }

    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
        angle -= sensitivity;
        isChange = true;
    }
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
        angle += sensitivity;
        isChange = true;
    }
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) {
        impulseForce.x *= 10;
    }

    // Wrap angle between -PI and PI
    angle = (float) UtilsAngles::WrapPosNegPI(angle);

    if (isChange) {
        glm::mat4 rotationMat(1);
        rotationMat = glm::rotate(rotationMat, -angle, glm::vec3(0.0f, 1.0f, 0.0f));

        impulseForce = glm::vec3(rotationMat * glm::vec4(impulseForce, 1.0f));

        ball->addForce(physx::PxVec3(impulseForce.x, impulseForce.y, impulseForce.z), physx::PxForceMode::eIMPULSE);
    }

}
