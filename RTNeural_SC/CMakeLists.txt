
set(CMAKE_CXX_STANDARD 17)

cmake_minimum_required (VERSION 3.5)
project (RTNeuralUGen)

include_directories(${SC_PATH}/include/plugin_interface)
include_directories(${SC_PATH}/include/common)
include_directories(${SC_PATH}/external_libraries/libsndfile/)

set(CMAKE_SHARED_MODULE_PREFIX "")
if(APPLE OR WIN32)
set(CMAKE_SHARED_MODULE_SUFFIX ".scx")
endif()

# add the plugin we are making
add_library(RTNeuralUGen MODULE RTNeuralUGen.cpp)

# add the submodule
add_subdirectory(../RTNeuralCPP ../build/RTNeuralCPP)

if (UNIX)
    message ("LINUX or MAC OS X")
    set(BUILD_TESTING OFF CACHE BOOL "Don't build libsamplerate tests!")
    add_subdirectory(../libsamplerate ../build/libsamplerate)
    message(STATUS "VERSION for libsamplerate: ${libsamplerate_version}")
    target_link_libraries(RTNeuralUGen ${CMAKE_CURRENT_SOURCE_DIR}/../libsamplerate/build/src/libsamplerate.a)
endif (UNIX)
if (WIN32)
    message ("Windows")
    target_link_libraries(RTNeuralUGen ${CMAKE_CURRENT_SOURCE_DIR}/../libsamplerate/lib/samplerate.lib)
endif (WIN32)

target_link_libraries(RTNeuralUGen RTNeural)

set_property(TARGET RTNeuralUGen PROPERTY CXX_STANDARD 17)

add_custom_command(
        TARGET ${PROJECT_NAME}  POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_CURRENT_BINARY_DIR}/RTNeuralUGen.scx
                ${CMAKE_CURRENT_BINARY_DIR}/../RTNeural_SC/RTNeuralUGen.scx)

# add_custom_command(TARGET ${PROJECT_NAME}  POST_BUILD
#         COMMAND ${CMAKE_COMMAND} -E copy_directory
#                 ${CMAKE_CURRENT_SOURCE_DIR}/../RTNeural_python
#                 ${CMAKE_CURRENT_BINARY_DIR}/../RTNeural/RTNeural_python)

# cmake -DCMAKE_BUILD_TYPE=Release -DSC_PATH='/Users/spluta1/Documents/SC/SuperColliderBuild/SuperCollider' ..
# cmake --build . --config Release