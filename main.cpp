#include <iostream>
#include "PxPhysicsAPI.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include "shaderClass.h"
#include "Camera.h"
#include "MyObjects.h"
#include "Sphere.h"

using namespace std;
using namespace physx;

static PxDefaultAllocator mallocator;
static PxDefaultErrorCallback merrorCallback;

#define WIDTH 800
#define HEIGHT 600
#define SAMPLES 8

int main() {

    // PhysX simulation
    //    PxDefaultCpuDispatcher* cpuDispatcher = PxDefaultCpuDispatcherCreate(1);
    PxFoundation *foundation = PxCreateFoundation(PX_PHYSICS_VERSION, mallocator, merrorCallback);

    PxPhysics *physics = PxCreatePhysics(PX_PHYSICS_VERSION, *foundation, PxTolerancesScale());
    PxSceneDesc sceneDesc(physics->getTolerancesScale());
    sceneDesc.gravity = PxVec3(0.0f, -9.81f, 0.0f);
    auto gDispatcher = PxDefaultCpuDispatcherCreate(2);
    sceneDesc.cpuDispatcher = gDispatcher;
    sceneDesc.filterShader = PxDefaultSimulationFilterShader;

    PxScene *scene = physics->createScene(sceneDesc);

    const float ballRadius = 0.25f;

    PxMaterial *material = physics->createMaterial(0.5f, 0.5f, 0.1f);
    PxTransform tableTransform(PxVec3(0.0f, 0.0f, 0.0f), PxQuat(PxIdentity));
    PxTransform ballTransform(PxVec3(0.0f, 1.0f, 0.0f), PxQuat(PxIdentity));
    PxBoxGeometry tableGeometry(PxVec3(10.0f, 0.0001f, 10.0f));
    PxSphereGeometry ballGeometry(ballRadius);
    PxRigidStatic *table = PxCreateStatic(*physics, tableTransform, tableGeometry, *material);
    PxRigidDynamic *ball = PxCreateDynamic(*physics, ballTransform, ballGeometry, *material, 1.0f);
    ball->setAngularDamping(0.5f);
    ball->setLinearVelocity(PxVec3(0.2f, 0.0f, 0.0f));

    scene->addActor(*table);
    scene->addActor(*ball);

    // OpenGL rendering
    glfwInit();

    // Tell GLFW what version of OpenGL we are using
    // In this case we are using OpenGL 3.3
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    // Only use this if you don't have a framebuffer
    glfwWindowHint(GLFW_SAMPLES, SAMPLES);
    // Tell GLFW we are using the CORE profile
    // So that means we only have the modern functions
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow *window = glfwCreateWindow(WIDTH, HEIGHT, "PhysX Table Simulation", nullptr, nullptr);
    glfwMakeContextCurrent(window);

    if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress)) {
        cout << "Failed to initialize GLAD" << endl;
        return -1;
    }

    // Generates Shader object using shaders default.vert and default.frag
    Shader shaderProgram("default.vert", "default.frag");

    std::vector<float> tableVertices = {
            // position                                            // color                      // normal
            -10.0f, ballRadius / 2.0f + 0.05f, -10.0f, 0.5f, 0.5f, 0.5f, 0.0f, 1.0f, 0.0f,
            10.0f, ballRadius / 2.0f + 0.05f, -10.0f, 0.5f, 0.5f, 0.5f, 0.0f, 1.0f, 0.0f,
            10.0f, ballRadius / 2.0f + 0.05f, 10.0f, 0.5f, 0.5f, 0.5f, 0.0f, 1.0f, 0.0f,
            -10.0f, ballRadius / 2.0f + 0.05f, 10.0f, 0.5f, 0.5f, 0.5f, 0.0f, 1.0f, 0.0f,
    };

    std::vector<unsigned int> tableIndices = {
            0, 1, 2,
            0, 2, 3
    };

    float ballColors[3] = {0.8f, 0.64f, 0.0f};

    auto tableObject = MyObjects(tableVertices, tableIndices);
    auto ballObject = Sphere(30, 30, ballRadius, ballColors);

    glEnable(GL_DEPTH_TEST);

    Camera camera(WIDTH, HEIGHT, glm::vec3(0.0f, 1.0f, 5.0f));

    while (!glfwWindowShouldClose(window)) {
        scene->simulate(1.0f / 60.0f);
        scene->fetchResults(true);

        PxVec3 ballPosition = ball->getGlobalPose().p;
        PxQuat ballRotation = ball->getGlobalPose().q;

        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        shaderProgram.Activate();

        // Handles camera inputs
        camera.Inputs(window);
        // Updates and exports the camera matrix to the Vertex Shader
        camera.Matrix(45.0f, 0.1f, 100.0f, shaderProgram, "camMatrix");
        glUniform3f(glGetUniformLocation(shaderProgram.ID, "camPos"), camera.Position.x, camera.Position.y, camera.Position.z);

        // render table
        tableObject.Draw(shaderProgram.ID);

        // render ball
        ballObject.Draw(shaderProgram.ID, ballPosition, ballRotation);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    tableObject.Delete();
    ballObject.Delete();
    shaderProgram.Delete();
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
