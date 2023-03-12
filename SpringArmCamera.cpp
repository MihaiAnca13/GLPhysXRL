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

void SpringArmCamera::Matrix(glm::vec3 objectPos, glm::mat3 objectRot, float FOVdeg, float nearPlane, float farPlane, Shader &shader, const char *uniform) {
    // Initializes matrices since otherwise they will be the null matrix
    glm::mat4 view = glm::mat4(1.0f);
    glm::mat4 projection = glm::mat4(1.0f);

    // update position based on objectTransform
    Position = objectRot * Position;
    Position = Position + objectPos - _lastObjPos;
    _lastObjPos = objectPos;

    // Makes camera look in the right direction from the right position
    view = glm::lookAt(Position, objectPos, Up);
    // Adds perspective to the scene
    projection = glm::perspective(glm::radians(FOVdeg), (float) width / height, nearPlane, farPlane);

    // Exports the camera matrix to the Vertex Shader
    glUniformMatrix4fv(glGetUniformLocation(shader.ID, uniform), 1, GL_FALSE, glm::value_ptr(projection * view));
}
