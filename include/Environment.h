//
// Created by mihai on 31/03/23.
//

#ifndef C_ML_ENVIRONMENT_H
#define C_ML_ENVIRONMENT_H

#include <iostream>
#include "PxPhysicsAPI.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include "shaderClass.h"
#include "SpringArmCamera.h"
#include "Table.h"
#include "Sphere.h"
#include "Shadow.h"
#include "Skybox.h"
#include "Model.h"

using namespace std;
using namespace physx;


class Environment {
public:
    static PxDefaultAllocator mallocator;
    static PxDefaultErrorCallback merrorCallback;

    int mWidth;
    int mHeight;
    float mBounds;
    float mBallDensity;

    glm::vec3 glmBallP{0.0f, 0.0f, 0.0f};
    glm::vec3 initialBallPos = glm::vec3(13.0f, 1.0f, 7.8f);
    PxQuat ballRotation;
    PxVec3 ballPosition;

    const float ballRadius = 1.0f;

    bool isOpen = true;
    bool springCamera = true;

    GLFWwindow *window;

    Shader shaderProgram{};
    Shader shadowMapProgram{};
    Shader skyboxShader{};
    Skybox skybox;
    Shadow shadowObject{};
    glm::mat4 lightProjection{};
    SpringArmCamera springArmCamera;
    Camera camera;

    PxFoundation *foundation;
    PxScene *scene;
    PxCooking *cooking;
    PxPhysics *physics;
    PxDefaultCpuDispatcher *gDispatcher;

    PxRigidDynamic *ball;

    Sphere ballObject;
    Model obstacleScene;

    Environment(int width, int height, float bounds, float ballDensity) : mWidth(width), mHeight(height), mBounds(bounds), mBallDensity(ballDensity) {
        Init();
    };

    void CleanUp();

    void Step();

    void Init();
};

#endif //C_ML_ENVIRONMENT_H
