//
// Created by mihai on 31/03/23.
//
#include "Environment.h"

PxDefaultAllocator Environment::mallocator;
PxDefaultErrorCallback Environment::merrorCallback;


void Environment::Init() {
    // PhysX simulation
    foundation = PxCreateFoundation(PX_PHYSICS_VERSION, mallocator, merrorCallback);

    physics = PxCreatePhysics(PX_PHYSICS_VERSION, *foundation, PxTolerancesScale());
    PxSceneDesc sceneDesc(physics->getTolerancesScale());
    sceneDesc.gravity = PxVec3(0.0f, -9.81f, 0.0f);
    gDispatcher = PxDefaultCpuDispatcherCreate(2);
    sceneDesc.cpuDispatcher = gDispatcher;
    sceneDesc.filterShader = PxDefaultSimulationFilterShader;
    PxCookingParams params(physics->getTolerancesScale());
    cooking = PxCreateCooking(PX_PHYSICS_VERSION, *foundation, params);
    scene = physics->createScene(sceneDesc);

    PxMaterial *material = physics->createMaterial(0.5f, 0.5f, 0.1f);
    PxTransform ballTransform(PxVec3(initialBallPos.x, initialBallPos.y, initialBallPos.z), PxQuat(PxIdentity));
    PxSphereGeometry ballGeometry(ballRadius / 4);
    ball = PxCreateDynamic(*physics, ballTransform, ballGeometry, *material, mBallDensity);
    ball->setAngularDamping(3.0f);
    scene->addActor(*ball);

    // OpenGL rendering
    glfwInit();

    // Tell GLFW what version of OpenGL we are using
    // In this case we are using OpenGL 3.3
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    // Tell GLFW we are using the CORE profile
    // So that means we only have the modern functions
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window = glfwCreateWindow(mWidth, mHeight, "PhysX C_ML Simulation", nullptr, nullptr);
    glfwMakeContextCurrent(window);

    if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress)) {
        cout << "Failed to initialize GLAD" << endl;
        throw std::invalid_argument("Failed to initialize GLAD");
    }

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    (void) io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    //ImGui::StyleColorsLight();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    // Generates Shader object using shaders default.vert and default.frag
    shaderProgram = Shader("default.vert", "default.frag");
    shadowMapProgram = Shader("shadowMap.vert", "shadowMap.frag");
    skyboxShader = Shader("skybox.vert", "skybox.frag");

    float ballColors[3] = {0.2f, 0.5f, 0.8f};
    ballObject = Sphere(30, 30, ballRadius, ballColors);
    obstacleScene = Model("resources/scene.obj");

    obstacleScene.addActorsToScene(physics, cooking, scene, material);
    physx::PxRigidBodyExt::updateMassAndInertia(*ball, mBallDensity);

    // Enables Depth Testing
    glEnable(GL_DEPTH_TEST);
    // Enables Multisampling
    glEnable(GL_MULTISAMPLE);
    // Enables Cull Facing
    glEnable(GL_CULL_FACE);
    // Uses counter clock-wise standard
    glFrontFace(GL_CCW);

    springArmCamera = SpringArmCamera(mWidth, mHeight, initialBallPos + glm::vec3(3.0f, 1.0f, 0.0f), initialBallPos);
    camera = Camera(mWidth, mHeight, glm::vec3(0.0f, 1.0f, 5.0f));

    shaderProgram.Activate();
    glm::vec3 lightPos = glm::vec3(1.0f, 1.0f, -0.8f);
    glUniform3f(glGetUniformLocation(shaderProgram.ID, "lightDirection"), lightPos.x, lightPos.y, lightPos.z);
    glUniform1i(glGetUniformLocation(shaderProgram.ID, "skybox"), 6);

    // cube map
    skyboxShader.Activate();
    glUniform1i(glGetUniformLocation(skyboxShader.ID, "skybox"), 6);

    // skybox
    skybox = Skybox(skyboxShader.ID);

    // Framebuffer for Shadow Map
    shadowObject = Shadow(4096, 4096);

    // Matrices needed for the light's perspective
    const float orthoDistance = 23.0f;
    glm::mat4 orthgonalProjection = glm::ortho(-orthoDistance, orthoDistance, -orthoDistance, orthoDistance, 0.1f, 175.0f);
    glm::mat4 lightView = glm::lookAt(50.0f * lightPos, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    lightProjection = orthgonalProjection * lightView;

    shadowMapProgram.Activate();
    glUniformMatrix4fv(glGetUniformLocation(shadowMapProgram.ID, "lightProjection"), 1, GL_FALSE, glm::value_ptr(lightProjection));
    glm::mat4 model = glm::mat4(1.0f);
    glUniformMatrix4fv(glGetUniformLocation(shadowMapProgram.ID, "model"), 1, GL_FALSE, glm::value_ptr(model));

    isOpen = true;
    springCamera = true;
}


void Environment::Step() {
    if (glfwWindowShouldClose(window)) {
        return;
    }

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

    ballPosition = ball->getGlobalPose().p;
    ballRotation = ball->getGlobalPose().q;

    glmBallP = glm::vec3(ballPosition.x, ballPosition.y, ballPosition.z);

    // check if ball position is out of bounds
    if (glmBallP.x > mBounds || glmBallP.x < -mBounds || glmBallP.y > mBounds || glmBallP.y < -mBounds || glmBallP.z > mBounds || glmBallP.z < -mBounds) {
        glmBallP = glm::clamp(glmBallP, glm::vec3(-mBounds), glm::vec3(mBounds));

        ball->setGlobalPose(PxTransform(glmBallP.x, glmBallP.y, glmBallP.z, ballRotation), true);
    }

    // Depth testing needed for Shadow Map
    glEnable(GL_DEPTH_TEST);

    // Preparations for the Shadow Map
    shadowMapProgram.Activate();

    shadowObject.bindFramebuffer();
    glClear(GL_DEPTH_BUFFER_BIT);

    glCullFace(GL_FRONT);

    // render ball
    ballObject.Draw(shadowMapProgram.ID, ballPosition, ballRotation);

    // render scene
    obstacleScene.Draw(shadowMapProgram.ID);

    glCullFace(GL_BACK);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // Render the scene normally
    shaderProgram.Activate();
    glViewport(0, 0, mWidth, mHeight);
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUniformMatrix4fv(glGetUniformLocation(shaderProgram.ID, "lightProjection"), 1, GL_FALSE, glm::value_ptr(lightProjection));


    // Updates and exports the camera matrix to the Vertex Shader
    if (springCamera) {
        springArmCamera.Inputs(window, ball);
        springArmCamera.Matrix(glmBallP, 45.0f, 1.6f, 100.0f, shaderProgram, "camMatrix");
        glUniform3f(glGetUniformLocation(shaderProgram.ID, "camPos"), springArmCamera.Position.x, springArmCamera.Position.y, springArmCamera.Position.z);
    } else {
        // Handles camera inputs
        camera.Inputs(window);
        // Updates the camera matrix
        camera.Matrix(45.0f, 0.1f, 100.0f, shaderProgram, "camMatrix");
        glUniform3f(glGetUniformLocation(shaderProgram.ID, "camPos"), camera.Position.x, camera.Position.y, camera.Position.z);
    }

    // Bind the Shadow Map to the Texture Unit 0
    shadowObject.bindTexture(shaderProgram.ID, 0);

    // render scene
    glUniform1ui(glGetUniformLocation(shaderProgram.ID, "specMulti"), 2);
    if (springCamera) {
        obstacleScene.Draw(shaderProgram.ID, glmBallP, springArmCamera.Position);
    } else {
        obstacleScene.Draw(shaderProgram.ID);
    }

    // render ball
    glUniform1ui(glGetUniformLocation(shaderProgram.ID, "specMulti"), 16);
    ballObject.Draw(shaderProgram.ID, ballPosition, ballRotation);

    // Since the cubemap will always have a depth of 1.0, we need that equal sign so it doesn't get discarded
    glDepthFunc(GL_LEQUAL);

    glFrontFace(GL_CW);
    skyboxShader.Activate();

    if (springCamera)
        skybox.Draw(springArmCamera, mWidth, mHeight);
    else
        skybox.Draw(camera, mWidth, mHeight);

    // Switch back to the normal depth function
    glDepthFunc(GL_LESS);
    glFrontFace(GL_CCW);

    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    glfwSwapBuffers(window);
    glfwPollEvents();
}

void Environment::CleanUp() {
    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();


    PX_RELEASE(cooking);
    PX_RELEASE(scene);
    PX_RELEASE(gDispatcher);
    PxCloseExtensions();
    PX_RELEASE(physics);
    PX_RELEASE(foundation);
//    if(gPvd)
//    {
//        PxPvdTransport* transport = gPvd->getTransport();
//        gPvd->release();	gPvd = NULL;
//        PX_RELEASE(transport);
//    }
//    PX_RELEASE(foundation);

    ballObject.Delete();
    obstacleScene.Delete();
    shaderProgram.Delete();
    shadowMapProgram.Delete();
    skyboxShader.Delete();
    glfwDestroyWindow(window);
    glfwTerminate();
}