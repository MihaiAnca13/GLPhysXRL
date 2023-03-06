#include <iostream>
#include "PxPhysicsAPI.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "shaderClass.h"
#include "Camera.h"
#include "MyObjects.h"

using namespace std;
using namespace physx;

static PxDefaultAllocator mallocator;
static PxDefaultErrorCallback merrorCallback;

#define WIDTH 800
#define HEIGHT 600

void sphereGeneration(unsigned int[], float[], int, int, float, const float *);


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
    GLFWwindow *window = glfwCreateWindow(WIDTH, HEIGHT, "PhysX Table Simulation", nullptr, nullptr);
    glfwMakeContextCurrent(window);

    if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress)) {
        cout << "Failed to initialize GLAD" << endl;
        return -1;
    }

    // Generates Shader object using shaders default.vert and default.frag
    Shader shaderProgram("default.vert", "default.frag");

    float tableVertices[] = {
            // position                                            // color                      // normal
            -10.0f, ballRadius / 2.0f + 0.05f, -10.0f, 0.5f, 0.5f, 0.5f, 0.0f, 1.0f, 0.0f,
            10.0f, ballRadius / 2.0f + 0.05f, -10.0f, 0.5f, 0.5f, 0.5f, 0.0f, 1.0f, 0.0f,
            10.0f, ballRadius / 2.0f + 0.05f, 10.0f, 0.5f, 0.5f, 0.5f, 0.0f, 1.0f, 0.0f,
            -10.0f, ballRadius / 2.0f + 0.05f, 10.0f, 0.5f, 0.5f, 0.5f, 0.0f, 1.0f, 0.0f,
    };

    unsigned int tableIndices[] = {
            0, 1, 2,
            0, 2, 3
    };

    const int numSlices = 30;
    const int numStacks = 30;
    const float ballColors[3] = {0.8f, 0.64f, 0.0f};

    unsigned int ballIndices[numSlices * numStacks * 6];
    float ballVertices[(numSlices + 1) * (numStacks + 1) * 9];

    sphereGeneration(ballIndices, ballVertices, numSlices, numStacks, ballRadius, ballColors);

    auto tableObject = MyObjects(tableVertices, tableIndices, sizeof(tableIndices) / sizeof(unsigned int));

    unsigned int ballVAO, ballVBO, ballEBO;
    glGenVertexArrays(1, &ballVAO);
    glGenBuffers(1, &ballVBO);
    glGenBuffers(1, &ballEBO);
    glBindVertexArray(ballVAO);
    glBindBuffer(GL_ARRAY_BUFFER, ballVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(ballVertices), ballVertices, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ballEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(ballIndices), ballIndices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void *) nullptr);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void *) (3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void *) (6 * sizeof(float)));
    glEnableVertexAttribArray(2);

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

        glm::mat4 model = glm::mat4(1.0f);

        // render ball
        glBindVertexArray(ballVAO);
        model = glm::mat4(1.0f);
        model = glm::translate(model, glm::vec3(ballPosition.x, ballPosition.y, ballPosition.z));
        glm::mat4 ballRotationMatrix = glm::mat4_cast(glm::quat(ballRotation.w, ballRotation.x, ballRotation.y, ballRotation.z));
        model = model * ballRotationMatrix;
        model = glm::scale(model, glm::vec3(0.25f, 0.25f, 0.25f));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram.ID, "model"), 1, GL_FALSE, glm::value_ptr(model));
        glDrawElements(GL_TRIANGLES, numSlices * numStacks * 6, GL_UNSIGNED_INT, nullptr);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    tableObject.Delete();
    shaderProgram.Delete();
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}


void sphereGeneration(unsigned int indices[], float vertices[], int numSlices = 30, int numStacks = 30, float radius = 1.0f, const float *color = nullptr) {
    int vertexIndex = 0;
    int indexIndex = 0;

    if (color == nullptr) {
        printf("No color provided!");
        throw (std::exception());
    }

    for (int stack = 0; stack <= numStacks; ++stack) {
        float phi = stack * PxPi / numStacks;
        float sinPhi = sin(phi);
        float cosPhi = cos(phi);
        for (int slice = 0; slice <= numSlices; ++slice) {
            float theta = slice * 2 * PxPi / numSlices;
            float sinTheta = sin(theta);
            float cosTheta = cos(theta);

            // Calculate position and normal
            vertices[vertexIndex++] = radius * sinPhi * cosTheta;
            vertices[vertexIndex++] = radius * sinPhi * sinTheta;
            vertices[vertexIndex++] = radius * cosPhi;

            // Add color to vertices
            vertices[vertexIndex++] = color[0];
            vertices[vertexIndex++] = color[1];
            vertices[vertexIndex++] = color[2];

            // Calculate normal
            float x = sinPhi * cosTheta;
            float y = sinPhi * sinTheta;
            float z = cosPhi;
            glm::vec3 normal(x, y, z);
            normal = glm::normalize(normal);

            // Add normal to vertices
            vertices[vertexIndex++] = normal.x;
            vertices[vertexIndex++] = normal.y;
            vertices[vertexIndex++] = normal.z;

            // Add indices
            if (stack != numStacks && slice != numSlices) {
                int nextStack = stack + 1;
                int nextSlice = slice + 1;
                indices[indexIndex++] = (stack * (numSlices + 1) + slice);
                indices[indexIndex++] = (nextStack * (numSlices + 1) + slice);
                indices[indexIndex++] = (nextStack * (numSlices + 1) + nextSlice);
                indices[indexIndex++] = (stack * (numSlices + 1) + slice);
                indices[indexIndex++] = (nextStack * (numSlices + 1) + nextSlice);
                indices[indexIndex++] = (stack * (numSlices + 1) + nextSlice);
            }
        }
    }
}