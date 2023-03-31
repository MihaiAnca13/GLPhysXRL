//
// Created by mihai on 08/03/23.
//

#ifndef C_ML_SHADOW_H
#define C_ML_SHADOW_H

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

class Shadow {
public:
    Shadow(unsigned int shadowMapWidth, unsigned int shadowMapHeight);
    Shadow() = default;
    unsigned int FBO, shadowMapWidth, shadowMapHeight, shadowMap;

    void bindFramebuffer() const;
    void bindTexture(GLuint shaderProgram, GLuint textureUnit) const;
};

#endif //C_ML_SHADOW_H
