#include <stdexcept>
#include "PxPhysicsAPI.h"

using namespace physx;

static PxDefaultErrorCallback gDefaultErrorCallback;
static PxDefaultAllocator gDefaultAllocatorCallback;


int main() {
    PxFoundation* mFoundation = PxCreateFoundation(PX_PHYSICS_VERSION, gDefaultAllocatorCallback,
                                     gDefaultErrorCallback);
    if(!mFoundation)
        throw std::runtime_error("PxCreateFoundation failed!");

    bool recordMemoryAllocations = true;

    PxPvd* mPvd = PxCreatePvd(*mFoundation);
    PxPvdTransport* transport = PxDefaultPvdSocketTransportCreate("127.0.0.1", 5425, 10);
    mPvd->connect(*transport,PxPvdInstrumentationFlag::eALL);

    PxPhysics* mPhysics = PxCreatePhysics(PX_PHYSICS_VERSION, *mFoundation,
                               PxTolerancesScale(), recordMemoryAllocations, mPvd);
    if(!mPhysics)
        throw std::runtime_error("PxCreatePhysics failed!");

    PxSceneDesc sceneDesc(mPhysics->getTolerancesScale());
    sceneDesc.gravity = PxVec3(0.0f, -9.81f, 0.0f);
    auto gDispatcher = PxDefaultCpuDispatcherCreate(2);
    sceneDesc.cpuDispatcher	= gDispatcher;
    sceneDesc.filterShader	= PxDefaultSimulationFilterShader;
    PxScene* gScene = mPhysics->createScene(sceneDesc);

    PxPvdSceneClient* pvdClient = gScene->getScenePvdClient();
    if(pvdClient)
    {
        pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_CONSTRAINTS, true);
        pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_CONTACTS, true);
        pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_SCENEQUERIES, true);
    }
    PxMaterial* mMaterial = mPhysics->createMaterial(0.5f, 0.5f, 0.6f);

    PxRigidStatic* groundPlane = PxCreatePlane(*mPhysics, PxPlane(0,1,0,0), *mMaterial);
    gScene->addActor(*groundPlane);

    return 0;
}