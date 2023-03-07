//
// Created by mihai on 06/03/23.
//

#ifndef C_ML_SPHERE_H
#define C_ML_SPHERE_H

#include "MyObjects.h"
#include "PxPhysicsAPI.h"

class Sphere : public MyObjects {
public:
    Sphere(int numSlices, int numStacks, float radius, float* colors);

    void Draw(unsigned int shaderProgram, const physx::PxVec3& position, const physx::PxQuat& rotation) const;

private:
    static void sphereGeneration(std::vector<unsigned int> &indices, std::vector<float> &vertices, int numSlices, int numStacks, float radius, const float *color);
};

#endif //C_ML_SPHERE_H
