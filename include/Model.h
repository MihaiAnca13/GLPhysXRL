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


class Model
{
public:
    explicit Model(const char *path)
    {
        loadModel(path);
    }

    void Draw(unsigned int shaderProgram) const;
    void Delete();
private:
    // model data
    std::vector<Mesh> meshes;
    std::string directory;

    void loadModel(const std::string& path);
    void processNode(aiNode *node, const aiScene *scene);
    Mesh processMesh(aiMesh *mesh, const aiScene *scene);
};

#endif //C_ML_MODEL_H
