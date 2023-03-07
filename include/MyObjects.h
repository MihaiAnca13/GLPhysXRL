//
// Created by mihai on 06/03/23.
//

#ifndef C_ML_MYOBJECTS_H
#define C_ML_MYOBJECTS_H

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glad/glad.h>
#include <vector>

class MyObjects {
public:
    MyObjects() = default;

    MyObjects(std::vector<float> vertices, std::vector<unsigned int> indices);

    void Draw(unsigned int shaderProgram) const;

    void Delete();

private:
    unsigned int VAO{}, VBO{}, EBO{};
    GLsizei _numIndices{};

protected:
    void Initialise(std::vector<float> vertices, std::vector<unsigned int> indices);
};


#endif //C_ML_MYOBJECTS_H
