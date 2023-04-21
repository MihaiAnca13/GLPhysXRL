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
#include <torch/torch.h>

using namespace std;
using namespace physx;
using namespace torch;

typedef struct {
    int width;
    int height;
    float bounds;
    float ballDensity;
    int numSubsteps;
    bool manualControl;
    bool headless;
    int maxSteps;
    float threshold;
    float bonusAchievedReward;
} Config;


typedef struct {
    PxVec3 ballPosition;
    PxVec3 ballVelocity;
    float direction;

    Tensor toTensor() {
        Tensor r = torch::zeros({7});
        r[0] = ballPosition.x;
        r[1] = ballPosition.y;
        r[2] = ballPosition.z;
        r[3] = ballVelocity.x;
        r[4] = ballVelocity.y;
        r[5] = ballVelocity.z;
        r[6] = direction;
        return r;
    }
} Observation;


typedef struct {
    Observation observation;
    double reward;
    bool done;
} StepResult;


class Environment {
public:
    static PxDefaultAllocator mallocator;
    static PxDefaultErrorCallback merrorCallback;

    int mWidth;
    int mHeight;
    float mBounds;
    float mBallDensity;
    int numSubsteps = 5;
    bool manualControl = false;
    bool headless = false;
    int maxSteps = 100;
    float threshold = 0.03f;
    float bonusAchievedReward = 10.0f;

    int _step = 0;

    PxVec3 goalPosition = PxVec3(-6.61703f, 1.31621f, -1.71782f);

    glm::vec3 glmBallP{0.0f, 0.0f, 0.0f};
    glm::vec3 initialBallPos = glm::vec3(13.0f, 0.3f, 7.8f);
    PxQuat ballRotation;
    PxVec3 ballPosition;

    float angle = 0.0f;
    float sensitivity = 0.0174533f * 2.0f; // 2 degrees
    float maxForce = 0.375f;

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

    Environment(Config config) {
        mWidth = config.width;
        mHeight = config.height;
        mBounds = config.bounds;
        mBallDensity = config.ballDensity;
        numSubsteps = config.numSubsteps;
        manualControl = config.manualControl;
        headless = config.headless;
        maxSteps = config.maxSteps;
        threshold = config.threshold;
        bonusAchievedReward = config.bonusAchievedReward;
        Init();
    };

    void CleanUp();

    void StepPhysics();

    StepResult Step(const Tensor& action);

    Observation Reset();

    Observation GetObservation();

    double ComputeReward();

    void Render();

    void Init();
};

#endif //C_ML_ENVIRONMENT_H
