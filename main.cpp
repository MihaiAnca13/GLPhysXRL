#include <iostream>
#include "PxPhysicsAPI.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "shaderClass.h"
#include "Camera.h"
#include "Table.h"
#include "Sphere.h"
#include "Shadow.h"

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

    const float ballRadius = 1.0f;
    const float tableSize = 30.0f;

    PxMaterial *material = physics->createMaterial(0.5f, 0.5f, 0.1f);
    PxTransform tableTransform(PxVec3(0.0f, 0.0f, 0.0f), PxQuat(PxIdentity));
    PxTransform ballTransform(PxVec3(0.0f, 1.0f, 0.0f), PxQuat(PxIdentity));
    PxBoxGeometry tableGeometry(PxVec3(tableSize / 8, 0.0001f, tableSize / 8));
    PxSphereGeometry ballGeometry(ballRadius / 4);
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
    Shader shadowMapProgram("shadowMap.vert", "shadowMap.frag");

    float ballColors[3] = {0.2f, 0.5f, 0.8f};
    float tableColors[3] = {0.5f, 0.5f, 0.5f};

    auto tableObject = Table(tableSize, 0.0f, tableColors);
    auto ballObject = Sphere(30, 30, ballRadius, ballColors);

    // Enables Depth Testing
    glEnable(GL_DEPTH_TEST);
    // Enables Multisampling
    glEnable(GL_MULTISAMPLE);

    Camera camera(WIDTH, HEIGHT, glm::vec3(0.0f, 1.0f, 5.0f));

    // Framebuffer for Shadow Map
    auto shadowObject = Shadow(2048, 2048);

    // Matrices needed for the light's perspective
    glm::vec3 lightPos = glm::vec3(1.0f, 1.0f, 0.0f);
    const float orthoDistance = 1.0f;
    glm::mat4 orthgonalProjection = glm::ortho(-orthoDistance, orthoDistance, -orthoDistance, orthoDistance, 0.1f, 75.0f);
    glm::mat4 lightView = glm::lookAt(10.0f * lightPos, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    glm::mat4 lightProjection = orthgonalProjection * lightView;

    shadowMapProgram.Activate();
    glUniformMatrix4fv(glGetUniformLocation(shadowMapProgram.ID, "lightProjection"), 1, GL_FALSE, glm::value_ptr(lightProjection));
    glm::mat4 model = glm::mat4(1.0f);
    glUniformMatrix4fv(glGetUniformLocation(shadowMapProgram.ID, "model"), 1, GL_FALSE, glm::value_ptr(model));

    glm::vec3 glmBallP;

    while (!glfwWindowShouldClose(window)) {
        scene->simulate(1.0f / 60.0f);
        scene->fetchResults(true);

        PxVec3 ballPosition = ball->getGlobalPose().p;
        PxQuat ballRotation = ball->getGlobalPose().q;

        PxVec3 tablePosition = table->getGlobalPose().p;

        // Depth testing needed for Shadow Map
        glEnable(GL_DEPTH_TEST);

        // Preparations for the Shadow Map
        shadowMapProgram.Activate();

        glmBallP = glm::vec3(ballPosition.x, ballPosition.y, ballPosition.z);
        lightView = glm::lookAt(10.0f * lightPos, glmBallP, glm::vec3(0.0f, 1.0f, 0.0f));
        lightProjection = orthgonalProjection * lightView;
        glUniformMatrix4fv(glGetUniformLocation(shadowMapProgram.ID, "lightProjection"), 1, GL_FALSE, glm::value_ptr(lightProjection));

        shadowObject.bindFramebuffer();
        glClear(GL_DEPTH_BUFFER_BIT);

        glCullFace(GL_FRONT);

        // render table
        tableObject.Draw(shadowMapProgram.ID, tablePosition);

        // render ball
        ballObject.Draw(shadowMapProgram.ID, ballPosition, ballRotation);

        glCullFace(GL_BACK);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        // Render the scene normally
        shaderProgram.Activate();
        glViewport(0, 0, WIDTH, HEIGHT);
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUniform3f(glGetUniformLocation(shaderProgram.ID, "lightDirection"), lightPos.x, lightPos.y, lightPos.z);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram.ID, "lightProjection"), 1, GL_FALSE, glm::value_ptr(lightProjection));

        // Handles camera inputs
        camera.Inputs(window);
        // Updates and exports the camera matrix to the Vertex Shader
        camera.Matrix(45.0f, 0.1f, 100.0f, shaderProgram, "camMatrix");
        glUniform3f(glGetUniformLocation(shaderProgram.ID, "camPos"), camera.Position.x, camera.Position.y, camera.Position.z);

        // Bind the Shadow Map to the Texture Unit 0
        shadowObject.bindTexture(shaderProgram.ID, 0);

        // render table
        tableObject.Draw(shaderProgram.ID, tablePosition);

        // render ball
        ballObject.Draw(shaderProgram.ID, ballPosition, ballRotation);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    tableObject.Delete();
    ballObject.Delete();
    shaderProgram.Delete();
    shadowMapProgram.Delete();
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
