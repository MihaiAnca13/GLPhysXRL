//
// Created by mihai on 31/03/23.
//
#include <cuda_runtime.h>
#include "Environment.h"

PxDefaultAllocator Environment::mallocator;
PxDefaultErrorCallback Environment::merrorCallback;


PxFilterFlags MyFilterShader(
        PxFilterObjectAttributes attributes0, PxFilterData filterData0,
        PxFilterObjectAttributes attributes1, PxFilterData filterData1,
        PxPairFlags& pairFlags, const void* constantBlock, PxU32 constantBlockSize)
{
    // let triggers through
    if(PxFilterObjectIsTrigger(attributes0) || PxFilterObjectIsTrigger(attributes1))
    {
        pairFlags = PxPairFlag::eTRIGGER_DEFAULT;
        return PxFilterFlag::eDEFAULT;
    }

    // generate contacts for all that were not filtered above
    pairFlags = PxPairFlag::eCONTACT_DEFAULT;

    // trigger the contact callback for pairs (A,B) where
    // the filtermask of A contains the ID of B and vice versa.
    if((filterData0.word0 & filterData1.word1) && (filterData1.word0 & filterData0.word1))
        pairFlags |= PxPairFlag::eNOTIFY_TOUCH_FOUND;

    // trigger a separation callback for pairs (A,B) where the collision group of A is included in the filtermask of B
    if(filterData0.word0 & filterData1.word1) {
        return PxFilterFlag::eKILL;
    }

    return PxFilterFlag::eDEFAULT;
}


Environment::Environment(EnvConfig config) {
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
    num_envs = config.num_envs;

    ballRotation.reserve(num_envs);
    ballPosition.reserve(num_envs);
    angle.reserve(num_envs);

    Init();
};


void Environment::Init() {
    // assert headless is true if manualControl is true, but allow for both to be false
    assert(!manualControl || headless);

    // PhysX simulation
    foundation = PxCreateFoundation(PX_PHYSICS_VERSION, mallocator, merrorCallback);
//    PxCudaContextManagerDesc cudaContextManagerDesc;
//    gCudaContextManager = PxCreateCudaContextManager(*foundation, cudaContextManagerDesc, PxGetProfilerCallback());
    physics = PxCreatePhysics(PX_PHYSICS_VERSION, *foundation, PxTolerancesScale());
    PxSceneDesc sceneDesc(physics->getTolerancesScale());
    sceneDesc.gravity = PxVec3(0.0f, -9.81f, 0.0f);
    gDispatcher = PxDefaultCpuDispatcherCreate(2);
    sceneDesc.cpuDispatcher = gDispatcher;
    sceneDesc.filterShader = MyFilterShader; //PxDefaultSimulationFilterShader;
//    sceneDesc.cudaContextManager = gCudaContextManager;
//    sceneDesc.flags |= PxSceneFlag::eENABLE_CCD;
//    sceneDesc.gpuMaxNumPartitions = 8;
//    sceneDesc.broadPhaseType = PxBroadPhaseType::eGPU;
    PxCookingParams params(physics->getTolerancesScale());
    cooking = PxCreateCooking(PX_PHYSICS_VERSION, *foundation, params);
    scene = physics->createScene(sceneDesc);

    PxMaterial *material = physics->createMaterial(0.5f, 0.5f, 0.1f);
    PxTransform ballTransform(PxVec3(initialBallPos.x, initialBallPos.y, initialBallPos.z), PxQuat(PxIdentity));
    PxSphereGeometry ballGeometry(ballRadius / 4);

    balls.reserve(num_envs);
    for (int i = 0; i < num_envs; i++) {
        auto ball = PxCreateDynamic(*physics, ballTransform, ballGeometry, *material, mBallDensity);
        ball->userData = new ActorUserData("ball" + std::to_string(i));
        ball->setAngularDamping(3.0f);

        // set collision filter data
        PxU32 numShapes = ball->getNbShapes();
        for (PxU32 j = 0; j < numShapes; j++) {
            PxShape *shape = nullptr;
            ball->getShapes(&shape, 1, j);
            if (shape != nullptr) {
                PxFilterData filterData;
                filterData.word0 = 1 << i;
                filterData.word1 =  ~(1 << i);
                shape->setSimulationFilterData(filterData);
            }
        }

        scene->addActor(*ball);
        balls.push_back(ball);
    }

    if (headless) {
        obstacleScene = Model("resources/scene.obj", headless);

        obstacleScene.addActorsToScene(physics, cooking, scene, material);

        // update mass and interia for all balls
        for (auto ball: balls) {
            physx::PxRigidBodyExt::updateMassAndInertia(*ball, mBallDensity);
        }

        return;
    }

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
    shaderProgram = Shader("shaders/default.vert", "shaders/default.frag");
    shadowMapProgram = Shader("shaders/shadowMap.vert", "shaders/shadowMap.frag");
    skyboxShader = Shader("shaders/skybox.vert", "shaders/skybox.frag");

    float ballColors[3] = {0.2f, 0.5f, 0.8f};
    ballObject = Sphere(30, 30, ballRadius, ballColors);
    obstacleScene = Model("resources/scene.obj");

    obstacleScene.addActorsToScene(physics, cooking, scene, material);

    // update mass and interia for all balls
    for (auto ball: balls) {
        physx::PxRigidBodyExt::updateMassAndInertia(*ball, mBallDensity);
    }

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


void Environment::StepPhysics() {
    scene->simulate(1.0f / 60.0f);
    scene->fetchResults(true);

    // update balls position
    for (int i = 0; i < num_envs; i++) {
        ballPosition[i] = balls[i]->getGlobalPose().p;
        ballRotation[i] = balls[i]->getGlobalPose().q;
    }
}


Tensor Environment::Reset() {
    // reset balls position
    for (int i = 0; i < num_envs; i++) {
        balls[i]->setLinearVelocity(PxVec3(0.0f, 0.0f, 0.0f));
        balls[i]->setAngularVelocity(PxVec3(0.0f, 0.0f, 0.0f));
        balls[i]->setGlobalPose(PxTransform(PxVec3(initialBallPos.x, initialBallPos.y, initialBallPos.z), PxQuat(PxIdentity)));
        angle[i] = 0.0f;
    }
    _step = 0;
    springArmCamera = SpringArmCamera(mWidth, mHeight, initialBallPos + glm::vec3(3.0f, 1.0f, 0.0f), initialBallPos);
    StepPhysics();
    return GetObservation();
}


Tensor Environment::GetObservation() {
    // create a tensor with shape {numBalls, observation_size}
    Tensor observation = torch::zeros({num_envs, observation_size}, floatOptions);

    for (int i = 0; i < num_envs; i++) {
        // create a GPU buffer for the data
        float *buffer;
        cudaMallocManaged(&buffer, observation_size * sizeof(float));

        // copy the data from CPU to GPU
        cudaMemcpy(buffer, &ballPosition[i].x, sizeof(float) * 3, cudaMemcpyHostToDevice);
        cudaMemcpy(buffer + 3, &ballRotation[i].x, sizeof(float) * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(buffer + 7, &angle[i], sizeof(float), cudaMemcpyHostToDevice);

        // create a tensor from the GPU buffer
        Tensor observation_tensor = torch::from_blob(buffer, {observation_size}, floatOptions);
        observation[i] = observation_tensor;

        cudaFree(buffer);
    }

    // fill it with ballPosition, ballRotation and angle
//    for (int i = 0; i < num_envs; i++) {
//        observation[i][0] = ballPosition[i].x;
//        observation[i][1] = ballPosition[i].y;
//        observation[i][2] = ballPosition[i].z;
//        observation[i][3] = ballRotation[i].x;
//        observation[i][4] = ballRotation[i].y;
//        observation[i][5] = ballRotation[i].z;
//        observation[i][6] = ballRotation[i].z;
//        observation[i][7] = angle[i];
//    }

    return observation;
}


StepResult Environment::Step(const Tensor &action) {
    // assert the shape of action is {numBalls, 2}
    assert(action.sizes() == torch::IntArrayRef({num_envs, 2}));

    for (int i = 0; i < num_envs; i++) {
        // clamp action between -1 and 1
        Tensor force = torch::clamp(action[i][0], -1.0f, 1.0f) * maxForce;
        Tensor rotation = torch::clamp(action[i][1], -1.0f, 1.0f);

        if (!manualControl) {
            // update angle using sensitivity
            angle[i] += rotation.item<float>() * sensitivity;

            auto fForce = force.item<float>();

            // apply force at given angle
            balls[i]->addForce(PxVec3(fForce * cos(angle[i]), 0.0f, fForce * sin(angle[i])), PxForceMode::eFORCE, true);
        }
    }

    // 5 substeps
    for (int i = 0; i < numSubsteps; i++) {
        // step physics
        StepPhysics();
        // render
        if (!headless) {
            Inputs();
            if (toRender)
                Render();
            else
                glfwPollEvents();
        }
    }

    for (int i = 0; i < num_envs; i++) {
        auto pos = ballPosition[i];
        // check if ball position is out of bounds in PhysX
        if (pos.x > mBounds || pos.x < -mBounds || pos.y > mBounds || pos.y < -mBounds || pos.z > mBounds || pos.z < -mBounds) {
            // clamp ball position to bounds
            pos.x = std::clamp(pos.x, -mBounds, mBounds);
            pos.y = std::clamp(pos.y, -mBounds, mBounds);
            pos.z = std::clamp(pos.z, -mBounds, mBounds);
            balls[i]->setGlobalPose(PxTransform(PxVec3(pos.x, pos.y, pos.z), ballRotation[i]), true);
        }
    }

    _step++;
    bool done = false;

    if (_step >= maxSteps) {
        done = true;
    }

    auto done_tensor = torch::ones({num_envs}, torch::kFloat32) * done;

    return {GetObservation(), ComputeReward(), done_tensor};
}


void Environment::Render() {
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

    if (springCamera) {
        glmBallP = glm::vec3(ballPosition[0].x, ballPosition[0].y, ballPosition[0].z);
    }

    // Depth testing needed for Shadow Map
    glEnable(GL_DEPTH_TEST);

    // Preparations for the Shadow Map
    shadowMapProgram.Activate();

    shadowObject.bindFramebuffer();
    glClear(GL_DEPTH_BUFFER_BIT);

    glCullFace(GL_FRONT);

    // render balls
    for (int i = 0; i < num_envs; i++) {
        ballObject.Draw(shadowMapProgram.ID, ballPosition[i], ballRotation[i]);
    }

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
        if (manualControl) {
            springArmCamera.Inputs(window, balls[0]);
            angle[0] = springArmCamera.angle;
        } else {
            springArmCamera.angle = angle[0];
        }

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

    // render balls
    glUniform1ui(glGetUniformLocation(shaderProgram.ID, "specMulti"), 16);
    for (int i = 0; i < num_envs; i++) {
        ballObject.Draw(shaderProgram.ID, ballPosition[i], ballRotation[i]);
    }

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
    PX_RELEASE(gCudaContextManager);
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


Tensor Environment::ComputeReward() {
    // initialize reward as a tensor of size num_envs
    Tensor reward = torch::zeros({num_envs}, torch::kFloat32);

    // compute reward as mean squared error between goal and ball position
    for (int i = 0; i < num_envs; i++) {
        double dist = 0.0f;
        for (int j = 0; j < 3; j++) {
            dist += pow(goalPosition[j] - ballPosition[i][j], 2);
        }
        dist = dist / 3.0f;

        reward[i] = -dist;
        // if within threshold of target, add bonus reward
        if (dist < threshold) {
            reward[i] += bonusAchievedReward;
        }
    }

    return reward;
}


void Environment::Inputs() {
    if (glfwGetKey(window, GLFW_KEY_V) == GLFW_PRESS) {
        if (!Vpressed) {
            toRender = !toRender;
        }
        Vpressed = true;
    }
    else {
        Vpressed = false;
    }
}