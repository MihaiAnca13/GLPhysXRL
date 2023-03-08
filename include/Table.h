//
// Created by mihai on 06/03/23.
//

#ifndef C_ML_TABLE_H
#define C_ML_TABLE_H

#include "MyObjects.h"
#include "PxPhysicsAPI.h"

class Table : public MyObjects {
public:
    Table(float size, float thickness, float* color);

    void Draw(unsigned int shaderProgram, const physx::PxVec3& position) const;

private:
    static void tableGeneration(std::vector<unsigned int> &indices, std::vector<float> &vertices, float size, float thickness, float* color);
};

#endif //C_ML_TABLE_H
