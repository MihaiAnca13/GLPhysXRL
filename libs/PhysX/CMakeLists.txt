# Include directories
include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/pvdruntime/include
        ${CMAKE_CURRENT_SOURCE_DIR}/source)

set(PhysXLIBS
        ${CMAKE_CURRENT_SOURCE_DIR}/bin/linux.clang/release/libPhysXExtensions_static_64.a
        ${CMAKE_CURRENT_SOURCE_DIR}/bin/linux.clang/release/libPhysX_static_64.a
        ${CMAKE_CURRENT_SOURCE_DIR}/bin/linux.clang/release/libPhysXPvdSDK_static_64.a
        ${CMAKE_CURRENT_SOURCE_DIR}/bin/linux.clang/release/libPhysXVehicle_static_64.a
        ${CMAKE_CURRENT_SOURCE_DIR}/bin/linux.clang/release/libPhysXCharacterKinematic_static_64.a
        ${CMAKE_CURRENT_SOURCE_DIR}/bin/linux.clang/release/libPhysXCooking_static_64.a
        ${CMAKE_CURRENT_SOURCE_DIR}/bin/linux.clang/release/libPhysXCommon_static_64.a
        ${CMAKE_CURRENT_SOURCE_DIR}/bin/linux.clang/release/libPhysXFoundation_static_64.a
        )

# add_library(PhysX STATIC)
add_library(PhysX INTERFACE)
target_link_libraries(PhysX INTERFACE ${PhysXLIBS})
target_include_directories(PhysX INTERFACE include)