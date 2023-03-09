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
#include "stb_image.h"

using namespace std;
using namespace physx;

static PxDefaultAllocator mallocator;
static PxDefaultErrorCallback merrorCallback;

#define WIDTH 800
#define HEIGHT 600
#define SAMPLES 8


float skyboxVertices[] =
        {
                //   Coordinates
                -1.0f, -1.0f,  1.0f,//        7--------6
                1.0f, -1.0f,  1.0f,//        /|       /|
                1.0f, -1.0f, -1.0f,//       4--------5 |
                -1.0f, -1.0f, -1.0f,//    | |      | |
                -1.0f,  1.0f,  1.0f,//   | 3------|-2
                1.0f,  1.0f,  1.0f,//    |/       |/
                1.0f,  1.0f, -1.0f,//    0--------1
                -1.0f,  1.0f, -1.0f
        };

unsigned int skyboxIndices[] =
        {
                // Right
                1, 2, 6,
                6, 5, 1,
                // Left
                0, 4, 7,
                7, 3, 0,
                // Top
                4, 5, 6,
                6, 7, 4,
                // Bottom
                0, 3, 2,
                2, 1, 0,
                // Back
                0, 1, 5,
                5, 4, 0,
                // Front
                3, 7, 6,
                6, 2, 3
        };



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

    Camera camera(WIDTH, HEIGHT, glm::vec3(0.0f, 1.0f, 5.0f));

    shaderProgram.Activate();
    glm::vec3 lightPos = glm::vec3(1.0f, 1.0f, -0.8f);
    glUniform3f(glGetUniformLocation(shaderProgram.ID, "lightDirection"), lightPos.x, lightPos.y, lightPos.z);

    // cube map
    skyboxShader.Activate();
    glUniform1i(glGetUniformLocation(skyboxShader.ID, "skybox"), 0);

    // Create VAO, VBO, and EBO for the skybox
    unsigned int skyboxVAO, skyboxVBO, skyboxEBO;
    glGenVertexArrays(1, &skyboxVAO);
    glGenBuffers(1, &skyboxVBO);
    glGenBuffers(1, &skyboxEBO);
    glBindVertexArray(skyboxVAO);
    glBindBuffer(GL_ARRAY_BUFFER, skyboxVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(skyboxVertices), &skyboxVertices, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, skyboxEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(skyboxIndices), &skyboxIndices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    // All the faces of the cubemap (make sure they are in this exact order)
    std::string facesCubemap[6] =
            {
                    "resources/skybox/px.png",
                    "resources/skybox/nx.png",
                    "resources/skybox/py.png",
                    "resources/skybox/ny.png",
                    "resources/skybox/pz.png",
                    "resources/skybox/nz.png"
            };

    // Creates the cubemap texture object
    unsigned int cubemapTexture;
    glGenTextures(1, &cubemapTexture);
    glBindTexture(GL_TEXTURE_CUBE_MAP, cubemapTexture);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    // These are very important to prevent seams
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    // This might help with seams on some systems
    //glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);

    // Cycles through all the textures and attaches them to the cubemap object
    for (unsigned int i = 0; i < 6; i++)
    {
        int width, height, nrChannels;
        unsigned char* data = stbi_load(facesCubemap[i].c_str(), &width, &height, &nrChannels, STBI_rgb);
        if (data)
        {
            stbi_set_flip_vertically_on_load(false);
            glTexImage2D
                    (
                            GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                            0,
                            GL_RGB,
                            width,
                            height,
                            0,
                            GL_RGB,
                            GL_UNSIGNED_BYTE,
                            data
                    );
            stbi_image_free(data);
        }
        else
        {
            std::cout << "Failed to load texture: " << facesCubemap[i] << std::endl;
            stbi_image_free(data);
        }
    }

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

        // Handles camera inputs
        camera.Inputs(window);
        // Updates and exports the camera matrix to the Vertex Shader
        camera.Matrix(45.0f, 0.1f, 100.0f, shaderProgram, "camMatrix");
        glUniform3f(glGetUniformLocation(shaderProgram.ID, "camPos"), camera.Position.x, camera.Position.y, camera.Position.z);

        // Bind the Shadow Map to the Texture Unit 0
        shadowObject.bindTexture(shaderProgram.ID, 0);

        // render table
        glUniform1ui(glGetUniformLocation(shaderProgram.ID, "specMulti"), 8);
        tableObject.Draw(shaderProgram.ID, tablePosition);

        // render ball
        glUniform1ui(glGetUniformLocation(shaderProgram.ID, "specMulti"), 16);
        ballObject.Draw(shaderProgram.ID, ballPosition, ballRotation);

        // Since the cubemap will always have a depth of 1.0, we need that equal sign so it doesn't get discarded
        glDepthFunc(GL_LEQUAL);

        skyboxShader.Activate();
        glm::mat4 view = glm::mat4(1.0f);
        glm::mat4 projection = glm::mat4(1.0f);
        // We make the mat4 into a mat3 and then a mat4 again in order to get rid of the last row and column
        // The last row and column affect the translation of the skybox (which we don't want to affect)
        view = glm::mat4(glm::mat3(glm::lookAt(camera.Position, camera.Position + camera.Orientation, camera.Up)));
        projection = glm::perspective(glm::radians(45.0f), (float)WIDTH / HEIGHT, 0.1f, 100.0f);
        glUniformMatrix4fv(glGetUniformLocation(skyboxShader.ID, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(skyboxShader.ID, "projection"), 1, GL_FALSE, glm::value_ptr(projection));

        // Draws the cubemap as the last object so we can save a bit of performance by discarding all fragments
        // where an object is present (a depth of 1.0f will always fail against any object's depth value)
        glBindVertexArray(skyboxVAO);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_CUBE_MAP, cubemapTexture);
        glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);

        // Switch back to the normal depth function
        glDepthFunc(GL_LESS);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    tableObject.Delete();
    ballObject.Delete();
    shaderProgram.Delete();
    shadowMapProgram.Delete();
    skyboxShader.Delete();
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
