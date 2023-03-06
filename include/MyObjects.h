//
// Created by mihai on 06/03/23.
//

#ifndef C_ML_MYOBJECTS_H
#define C_ML_MYOBJECTS_H

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include<glad/glad.h>


class MyObjects {
public:
    MyObjects(float vertices[], unsigned int indices[], GLsizei numIndices);

    void Draw(unsigned int shaderProgram) const;

    void Delete();

private:
    unsigned int VAO{}, VBO{}, EBO{};
    GLsizei _numIndices{};
};


#endif //C_ML_MYOBJECTS_H
