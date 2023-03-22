//
// Created by mihai on 17/03/23.
//

#ifndef C_ML_MODEL_H
#define C_ML_MODEL_H

#include "Mesh.h"
#include <vector>
#include <cstring>
#include "assimp/Importer.hpp"
#include "assimp/scene.h"
#include "assimp/postprocess.h"
#include <PxPhysicsAPI.h>


class Model {
public:
    explicit Model(const char *path) {
        loadModel(path);
    }

    void Draw(unsigned int shaderProgram) const;
    void Draw(unsigned int shaderProgram, glm::vec3 ballPosition, glm::vec3 cameraPosition);

    void Delete();

    void addActorsToScene(physx::PxPhysics *physics, physx::PxCooking *cooking, physx::PxScene *scene, physx::PxMaterial *material);

private:
    // model data
    std::vector<Mesh> meshes;
    std::string directory;

    void loadModel(const std::string &path);

    void processNode(aiNode *node, const aiScene *scene);

    Mesh processMesh(aiMesh *mesh, const aiScene *scene);

    physx::PxTriangleMesh *createTriangleMesh(physx::PxPhysics *physics, physx::PxCooking *cooking, const std::vector<Vertex> &vertices, const std::vector<unsigned int> &indices);
};

#endif //C_ML_MODEL_H
