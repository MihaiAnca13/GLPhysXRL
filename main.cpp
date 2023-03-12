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

using namespace std;
using namespace physx;

static PxDefaultAllocator mallocator;
static PxDefaultErrorCallback merrorCallback;

#define WIDTH 800
#define HEIGHT 600
#define SAMPLES 8
#define BOUNDS 100.0f


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

    glm::vec3 initialBallPos = glm::vec3(0.0f, 1.0f, 0.0f);

    PxMaterial *material = physics->createMaterial(0.5f, 0.5f, 0.1f);
    PxTransform tableTransform(PxVec3(0.0f, 0.0f, 0.0f), PxQuat(PxIdentity));
    PxTransform ballTransform(PxVec3(initialBallPos.x, initialBallPos.y, initialBallPos.z), PxQuat(PxIdentity));
    PxBoxGeometry tableGeometry(PxVec3(tableSize / 8, 0.0001f, tableSize / 8));
    PxSphereGeometry ballGeometry(ballRadius / 4);
    PxRigidStatic *table = PxCreateStatic(*physics, tableTransform, tableGeometry, *material);
    PxRigidDynamic *ball = PxCreateDynamic(*physics, ballTransform, ballGeometry, *material, 1.0f);
    ball->setAngularDamping(0.5f);
//    ball->setLinearVelocity(PxVec3(0.2f, 0.0f, 0.0f));
    ball->setAngularVelocity(PxVec3(0.0f, 0.0f, 8.0f));

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

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    //ImGui::StyleColorsLight();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    // Generates Shader object using shaders default.vert and default.frag
    Shader shaderProgram("default.vert", "default.frag");
    Shader shadowMapProgram("shadowMap.vert", "shadowMap.frag");
    Shader skyboxShader("skybox.vert", "skybox.frag");

    float ballColors[3] = {0.2f, 0.5f, 0.8f};
    float tableColors[3] = {0.3373f, 0.4902f, 0.2745f};

    auto tableObject = Table(tableSize, 0.0f, tableColors);
    auto ballObject = Sphere(30, 30, ballRadius, ballColors);

    // Enables Depth Testing
    glEnable(GL_DEPTH_TEST);
    // Enables Multisampling
    glEnable(GL_MULTISAMPLE);
    // Enables Cull Facing
//    glEnable(GL_CULL_FACE);
//    glCullFace(GL_FRONT);
    // Uses counter clock-wise standard
    glFrontFace(GL_CCW);

    SpringArmCamera springArmCamera(WIDTH, HEIGHT, initialBallPos + glm::vec3(-3.0f, 2.0f, 0.0f), initialBallPos);
    Camera camera(WIDTH, HEIGHT, glm::vec3(0.0f, 1.0f, 5.0f));

    shaderProgram.Activate();
    glm::vec3 lightPos = glm::vec3(1.0f, 1.0f, -0.8f);
    glUniform3f(glGetUniformLocation(shaderProgram.ID, "lightDirection"), lightPos.x, lightPos.y, lightPos.z);
    glUniform1i(glGetUniformLocation(shaderProgram.ID, "skybox"), 6);

    // cube map
    skyboxShader.Activate();
    glUniform1i(glGetUniformLocation(skyboxShader.ID, "skybox"), 6);

    // skybox
    auto skybox = Skybox(skyboxShader.ID);

    // Framebuffer for Shadow Map
    auto shadowObject = Shadow(2048, 2048);

    // Matrices needed for the light's perspective
    const float orthoDistance = 1.0f;
    glm::mat4 orthgonalProjection = glm::ortho(-orthoDistance, orthoDistance, -orthoDistance, orthoDistance, 0.1f, 75.0f);
    glm::mat4 lightView = glm::lookAt(10.0f * lightPos, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    glm::mat4 lightProjection = orthgonalProjection * lightView;

    shadowMapProgram.Activate();
    glUniformMatrix4fv(glGetUniformLocation(shadowMapProgram.ID, "lightProjection"), 1, GL_FALSE, glm::value_ptr(lightProjection));
    glm::mat4 model = glm::mat4(1.0f);
    glUniformMatrix4fv(glGetUniformLocation(shadowMapProgram.ID, "model"), 1, GL_FALSE, glm::value_ptr(model));

    glm::vec3 glmBallP;
    glm::mat3 glmBallR;
    PxQuat ballRotation = ball->getGlobalPose().q;
    float lastBallAngle = glm::eulerAngles(glm::quat(ballRotation.w, ballRotation.x, ballRotation.y, ballRotation.z)).y;

    bool isOpen = true;
    bool springCamera = true;
    while (!glfwWindowShouldClose(window)) {
        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::Begin("Debug window", &isOpen);
        ImGui::Checkbox("Spring Camera", &springCamera);
        ImGui::End();

        ImGui::Render();

        scene->simulate(1.0f / 60.0f);
        scene->fetchResults(true);

        PxVec3 ballPosition = ball->getGlobalPose().p;
        ballRotation = ball->getGlobalPose().q;

        PxVec3 tablePosition = table->getGlobalPose().p;

        // Depth testing needed for Shadow Map
        glEnable(GL_DEPTH_TEST);

        // Preparations for the Shadow Map
        shadowMapProgram.Activate();

        glmBallP = glm::vec3(ballPosition.x, ballPosition.y, ballPosition.z);

        float angle = glm::eulerAngles(glm::quat(ballRotation.w, ballRotation.x, ballRotation.y, ballRotation.z)).y;
        glmBallR = glm::mat3_cast(glm::angleAxis(lastBallAngle - angle, glm::vec3(0, 1, 0)));
        lastBallAngle = angle;

        // check if ball position is out of bounds
        if (glmBallP.x > BOUNDS || glmBallP.x < -BOUNDS || glmBallP.y > BOUNDS || glmBallP.y < -BOUNDS || glmBallP.z > BOUNDS || glmBallP.z < -BOUNDS) {
            glmBallP = glm::clamp(glmBallP, glm::vec3(-BOUNDS), glm::vec3(BOUNDS));

            ball->setGlobalPose(PxTransform(glmBallP.x, glmBallP.y, glmBallP.z, ballRotation), true);
        }

        lightView = glm::lookAt(10.0f * lightPos, glmBallP, glm::vec3(0.0f, 1.0f, 0.0f));
        lightProjection = orthgonalProjection * lightView;
        glUniformMatrix4fv(glGetUniformLocation(shadowMapProgram.ID, "lightProjection"), 1, GL_FALSE, glm::value_ptr(lightProjection));

        shadowObject.bindFramebuffer();
        glClear(GL_DEPTH_BUFFER_BIT);

        // render table
        tableObject.Draw(shadowMapProgram.ID, tablePosition);

        // render ball
        ballObject.Draw(shadowMapProgram.ID, ballPosition, ballRotation);

        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        // Render the scene normally
        shaderProgram.Activate();
        glViewport(0, 0, WIDTH, HEIGHT);
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUniformMatrix4fv(glGetUniformLocation(shaderProgram.ID, "lightProjection"), 1, GL_FALSE, glm::value_ptr(lightProjection));

        // Updates and exports the camera matrix to the Vertex Shader
        if (springCamera) {
            springArmCamera.Matrix(glmBallP, glmBallR, 45.0f, 0.1f, 100.0f, shaderProgram, "camMatrix");
            glUniform3f(glGetUniformLocation(shaderProgram.ID, "camPos"), springArmCamera.Position.x, springArmCamera.Position.y, springArmCamera.Position.z);
        }
        else {
            // Handles camera inputs
            camera.Inputs(window);
            // Updates the camera matrix
            camera.Matrix(45.0f, 0.1f, 100.0f, shaderProgram, "camMatrix");
            glUniform3f(glGetUniformLocation(shaderProgram.ID, "camPos"), camera.Position.x, camera.Position.y, camera.Position.z);
        }

        // Bind the Shadow Map to the Texture Unit 0
        shadowObject.bindTexture(shaderProgram.ID, 0);

        // render table
        glUniform1ui(glGetUniformLocation(shaderProgram.ID, "specMulti"), 8);
        tableObject.Draw(shaderProgram.ID, tablePosition);

        // render ball
        glUniform1ui(glGetUniformLocation(shaderProgram.ID, "specMulti"), 16);
        ballObject.Draw(shaderProgram.ID, ballPosition, ballRotation);

        // Since the cubemap will alwdys have a depth of 1.0, we need that equal sign so it doesn't get discarded
        glDepthFunc(GL_LEQUAL);

        skyboxShader.Activate();

        if(springCamera)
            skybox.Draw(springArmCamera, WIDTH, HEIGHT);
        else
            skybox.Draw(camera, WIDTH, HEIGHT);

        // Switch back to the normal depth function
        glDepthFunc(GL_LESS);

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    tableObject.Delete();
    ballObject.Delete();
    shaderProgram.Delete();
    shadowMapProgram.Delete();
    skyboxShader.Delete();
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
