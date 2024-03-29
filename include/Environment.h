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
#include <tensorboard_logger.h>

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
    int num_envs;
    float actionPenalty;
} EnvConfig;


typedef struct {
    Tensor observation;
    Tensor reward;
    Tensor done;
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
    float threshold = 0.1f;
    float bonusAchievedReward = 1.0f;
    int num_envs = 1;
    float actionPenalty = 0.001;

    int observation_size = 4;
    int action_size = 2;

    bool toRender = true;
    bool Vpressed = false;

    int _step = 0;

    Tensor goalPosition = torch::tensor({{-9.9f, 0.35712f, 1.6f}}, torch::TensorOptions().dtype(torch::kFloat32).device(kCUDA)); // -6.61703f, 1.31621f, -1.71782f

    glm::vec3 glmBallP{0.0f, 0.0f, 0.0f};
    glm::vec3 initialBallPos = glm::vec3(9.6f, 0.3f, 0.0f);  //glm::vec3(13.0f, 0.3f, 7.8f);

    Tensor ballRotation;
    Tensor ballPosition;
    Tensor angle;

    float sensitivity = 0.0174533f * 2.0f; // 2 degrees
    float maxForce = 0.375f;

    const float ballRadius = 1.0f;

    bool isOpen = true;
    bool springCamera = true;

    int _episode = 0;
    Tensor total_reward;
    float last_reward_mean = INT_MIN;

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
    PxCudaContextManager *gCudaContextManager;

    std::vector<PxRigidDynamic *> balls;

    TensorOptions floatOptions = torch::TensorOptions().dtype(torch::kFloat32).device(kCUDA).layout(torch::kStrided).requires_grad(false);

    Sphere ballObject;
    Model obstacleScene;

    Environment(EnvConfig config);

    void CleanUp();

    void StepPhysics(bool updateValues);

    void Inputs();

    StepResult Step(const Tensor &action, TensorBoardLogger *logger);

    Tensor Reset();

    Tensor GetObservation();

    Tensor ComputeReward(const Tensor& action);

    void Render();

    void Init();
};


class ActorUserData {
public:
    ActorUserData(const std::string &name) : m_name(name) {}

    const std::string &getName() const { return m_name; }

private:
    std::string m_name;
};

#endif //C_ML_ENVIRONMENT_H
