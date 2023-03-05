#include <iostream>
#include "PxPhysicsAPI.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

using namespace std;
using namespace physx;

static PxDefaultAllocator mallocator;
static PxDefaultErrorCallback merrorCallback;

void sphereGeneration(unsigned int[], float[], int, int, float, const float*);


int main() {

    // PhysX simulation
//    PxDefaultCpuDispatcher* cpuDispatcher = PxDefaultCpuDispatcherCreate(1);
    PxFoundation* foundation = PxCreateFoundation(PX_PHYSICS_VERSION, mallocator, merrorCallback);

    PxPhysics* physics = PxCreatePhysics(PX_PHYSICS_VERSION, *foundation, PxTolerancesScale());
    PxSceneDesc sceneDesc(physics->getTolerancesScale());
    sceneDesc.gravity = PxVec3(0.0f, -9.81f, 0.0f);
    auto gDispatcher = PxDefaultCpuDispatcherCreate(2);
    sceneDesc.cpuDispatcher = gDispatcher;
    sceneDesc.filterShader = PxDefaultSimulationFilterShader;

    PxScene* scene = physics->createScene(sceneDesc);

    PxMaterial* material = physics->createMaterial(0.5f, 0.5f, 0.1f);
    PxTransform tableTransform(PxVec3(0.0f, -1.0f, 0.0f), PxQuat(PxIdentity));
    PxTransform ballTransform(PxVec3(0.0f, 1.0f, 0.0f), PxQuat(PxIdentity));
    PxBoxGeometry tableGeometry(PxVec3(10.0f, 0.5f, 10.0f));
    PxSphereGeometry ballGeometry(0.25f);
    PxRigidStatic* table = PxCreateStatic(*physics, tableTransform, tableGeometry, *material);
    PxRigidDynamic* ball = PxCreateDynamic(*physics, ballTransform, ballGeometry, *material, 1.0f);
    ball->setAngularDamping(0.5f);
    ball->setLinearVelocity(PxVec3(0.2f, 0.0f, 0.0f));

    scene->addActor(*table);
    scene->addActor(*ball);

    // OpenGL rendering
    glfwInit();
    GLFWwindow* window = glfwCreateWindow(800, 600, "PhysX Table Simulation", nullptr, nullptr);
    glfwMakeContextCurrent(window);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        cout << "Failed to initialize GLAD" << endl;
        return -1;
    }

    const char* vertexShaderSource = "#version 330 core\n"
        "layout (location = 0) in vec3 aPos;\n"
        "layout (location = 1) in vec3 aColor;\n"
        "out vec3 vertexColor;\n"
        "uniform mat4 model;\n"
        "uniform mat4 view;\n"
        "uniform mat4 projection;\n"
        "void main()\n"
        "{\n"
        "   gl_Position = projection * view * model * vec4(aPos, 1.0);\n"
        "   vertexColor = aColor;\n"
        "}\0";

    const char* fragmentShaderSource = "#version 330 core\n"
        "in vec3 vertexColor;\n"
        "out vec4 FragColor;\n"
        "void main()\n"
        "{\n"
        "   FragColor = vec4(vertexColor, 1.0f);\n"
        "}\n\0";

    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
    glCompileShader(vertexShader);
    // check for compilation errors

    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
    glCompileShader(fragmentShader);
    // check for compilation errors

    unsigned int shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    // check for linking errors

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    float tableVertices[] = {
        // position                         // color
        -10.0f, -0.5f, -10.0f,   0.5f, 0.5f, 0.5f,
         10.0f, -0.5f, -10.0f,   0.5f, 0.5f, 0.5f,
         10.0f, -0.5f,  10.0f,   0.5f, 0.5f, 0.5f,
        -10.0f, -0.5f,  10.0f,   0.5f, 0.5f, 0.5f
    };

    unsigned int tableIndices[] = {
        0, 1, 2,
        0, 2, 3
    };

    const int numSlices = 30;
    const int numStacks = 30;

    unsigned int ballIndices[numSlices * numStacks * 6];
    float ballVertices[(numSlices + 1) * (numStacks + 1) * 6];

    const float ballColors[3] = { 0.8f, 0.64f, 0.0f};
    sphereGeneration(ballIndices, ballVertices, numSlices, numStacks, 1.0f, ballColors);

    unsigned int tableVAO, tableVBO, tableEBO;
    glGenVertexArrays(1, &tableVAO);
    glGenBuffers(1, &tableVBO);
    glGenBuffers(1, &tableEBO);
    glBindVertexArray(tableVAO);
    glBindBuffer(GL_ARRAY_BUFFER, tableVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(tableVertices), tableVertices, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, tableEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(tableIndices), tableIndices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)nullptr);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    unsigned int ballVAO, ballVBO, ballEBO;
    glGenVertexArrays(1, &ballVAO);
    glGenBuffers(1, &ballVBO);
    glGenBuffers(1, &ballEBO);
    glBindVertexArray(ballVAO);
    glBindBuffer(GL_ARRAY_BUFFER, ballVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(ballVertices), ballVertices, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ballEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(ballIndices), ballIndices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)nullptr);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glm::mat4 projection = glm::perspective(glm::radians(45.0f), 800.0f / 600.0f, 0.1f, 100.0f);
    glm::mat4 view = glm::lookAt(glm::vec3(0.0f, 3.0f, 5.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));

    while (!glfwWindowShouldClose(window)) {
        scene->simulate(1.0f / 60.0f);
        scene->fetchResults(true);

        PxVec3 ballPosition = ball->getGlobalPose().p;
        PxQuat ballRotation = ball->getGlobalPose().q;

        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(shaderProgram);

        // render table
        glBindVertexArray(tableVAO);
        glm::mat4 model = glm::mat4(1.0f);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);

        // render ball
        glBindVertexArray(ballVAO);
        model = glm::mat4(1.0f);
        model = glm::translate(model, glm::vec3(ballPosition.x, ballPosition.y, ballPosition.z));
        glm::mat4 ballRotationMatrix = glm::mat4_cast(glm::quat(ballRotation.w, ballRotation.x, ballRotation.y, ballRotation.z));
        model = model * ballRotationMatrix;
        model = glm::scale(model, glm::vec3(0.25f, 0.25f, 0.25f));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
        glDrawElements(GL_TRIANGLES, numSlices * numStacks * 6, GL_UNSIGNED_INT, nullptr);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();

    return 0;
}


void sphereGeneration(unsigned int indices[], float vertices[], int numSlices = 30, int numStacks = 30, float radius = 1.0f, const float *color = nullptr) {
    int vertexIndex = 0;
    int indexIndex = 0;

    for (int stack = 0; stack <= numStacks; ++stack)
    {
        float phi = stack * PxPi / numStacks;
        for (int slice = 0; slice <= numSlices; ++slice)
        {
            float theta = slice * 2 * PxPi / numSlices;
            vertices[vertexIndex++] = radius * sin(phi) * cos(theta);
            vertices[vertexIndex++] = radius * sin(phi) * sin(theta);
            vertices[vertexIndex++] = radius * cos(phi);
            vertices[vertexIndex++] = color[0];
            vertices[vertexIndex++] = color[1];
            vertices[vertexIndex++] = color[2];

            if (stack != numStacks && slice != numSlices)
            {
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