//
// Created by mihai on 17/03/23.
//
#include "Model.h"


void Model::loadModel(const std::string &path) {
    Assimp::Importer import;
    const aiScene *scene = import.ReadFile(path, aiProcess_Triangulate | aiProcess_FlipUVs);

    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
        std::cout << "ERROR::ASSIMP::" << import.GetErrorString() << std::endl;
        return;
    }
    directory = path.substr(0, path.find_last_of('/'));

    processNode(scene->mRootNode, scene);
}


physx::PxTriangleMesh* Model::getTriangleMesh(physx::PxPhysics* physics, physx::PxCooking* cooking) {
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;

    for (const Mesh& mesh : meshes)
    {
        size_t vertexOffset = vertices.size();
        vertices.insert(vertices.end(), mesh.vertices.begin(), mesh.vertices.end());

        for (unsigned int index : mesh.indices)
            indices.push_back((unsigned int)(vertexOffset + index));
    }

    return createTriangleMesh(physics, cooking, vertices, indices);
}


physx::PxTriangleMesh* Model::createTriangleMesh(physx::PxPhysics* physics, physx::PxCooking* cooking, const std::vector<Vertex>& vertices, const std::vector<unsigned int>& indices)
{
    // Convert vertices to PxVec3
    std::vector<physx::PxVec3> pxVertices(vertices.size());
    for (size_t i = 0; i < vertices.size(); ++i)
        pxVertices[i] = physx::PxVec3(vertices[i].Position.x, vertices[i].Position.y, vertices[i].Position.z);

    // Convert indices to PxU32
    std::vector<physx::PxU32> pxIndices(indices.begin(), indices.end());

    physx::PxTriangleMeshDesc meshDesc;
    meshDesc.points.count = (physx::PxU32)pxVertices.size();
    meshDesc.points.stride = sizeof(physx::PxVec3);
    meshDesc.points.data = pxVertices.data();

    meshDesc.triangles.count = (physx::PxU32)(pxIndices.size() / 3);
    meshDesc.triangles.stride = 3 * sizeof(physx::PxU32);
    meshDesc.triangles.data = pxIndices.data();

    physx::PxDefaultMemoryOutputStream writeBuffer;
    bool status = cooking->cookTriangleMesh(meshDesc, writeBuffer);
    if (!status)
        return nullptr;

    physx::PxDefaultMemoryInputData readBuffer(writeBuffer.getData(), writeBuffer.getSize());
    return physics->createTriangleMesh(readBuffer);
}


void Model::processNode(aiNode *node, const aiScene *scene) {
    // process all the node's meshes (if any)
    for (unsigned int i = 0; i < node->mNumMeshes; i++) {
        aiMesh *mesh = scene->mMeshes[node->mMeshes[i]];
        meshes.push_back(processMesh(mesh, scene));
    }
    // then do the same for each of its children
    for (unsigned int i = 0; i < node->mNumChildren; i++) {
        processNode(node->mChildren[i], scene);
    }
}


Mesh Model::processMesh(aiMesh *mesh, const aiScene *scene) {
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;
    aiMaterial *material = nullptr;

    if(mesh->mMaterialIndex >= 0)
    {
        material = scene->mMaterials[mesh->mMaterialIndex];
    }

    for (unsigned int i = 0; i < mesh->mNumVertices; i++) {
        Vertex vertex{};
        // process vertex positions, normals, color and reflectivity
        glm::vec3 vector;
        vector.x = mesh->mVertices[i].x;
        vector.y = mesh->mVertices[i].y;
        vector.z = mesh->mVertices[i].z;
        vertex.Position = vector;
        vector.x = mesh->mNormals[i].x;
        vector.y = mesh->mNormals[i].y;
        vector.z = mesh->mNormals[i].z;
        vertex.Normal = vector;
        if (material != nullptr) {
            aiColor4D diffuse;
            if (AI_SUCCESS == aiGetMaterialColor(material, AI_MATKEY_COLOR_DIFFUSE, &diffuse))
                vertex.Color = glm::vec3(diffuse.r, diffuse.g, diffuse.b);
        }
        else {
            vertex.Color = glm::vec3(0.1373f, 0.2235f, 0.3647f); // dark blue
        }
        vertex.Reflectivity = 0.0f;
        vertices.push_back(vertex);
    }
    // process indices
    for (unsigned int i = 0; i < mesh->mNumFaces; i++) {
        aiFace face = mesh->mFaces[i];
        for (unsigned int j = 0; j < face.mNumIndices; j++)
            indices.push_back(face.mIndices[j]);
    }

    return {vertices, indices};
}


void Model::Draw(unsigned int shaderProgram) const {
    for (const auto &mesh: meshes)
        mesh.Draw(shaderProgram);
}


void Model::Delete() {
    for (auto &mesh: meshes)
        mesh.Delete();
}