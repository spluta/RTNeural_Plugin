
set(CMAKE_CXX_STANDARD 17)

cmake_minimum_required (VERSION 3.5)
project (RTNeural)

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
add_subdirectory(../RTNeuralCPP ../RTNeuralCPP)

# target_include_directories(RTNeuralUGen PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/r8brain")

if (UNIX)
    message ("LINUX or MAC OS X")
    set(BUILD_TESTING OFF CACHE BOOL "Don't build libsamplerate tests!")
    add_subdirectory(../libsamplerate ../libsamplerate)
    message(STATUS "VERSION for libsamplerate: ${libsamplerate_version}")
    target_link_libraries(RTNeuralUGen ${CMAKE_CURRENT_SOURCE_DIR}/../libsamplerate/build/src/libsamplerate.a)
endif (UNIX)
if (WIN32)
    message ("Windows")
    target_link_libraries(RTNeuralUGen ${CMAKE_CURRENT_SOURCE_DIR}/../libsamplerate/lib/samplerate.lib)
endif (WIN32)

target_link_libraries(RTNeuralUGen RTNeural)

set_property(TARGET RTNeuralUGen PROPERTY CXX_STANDARD 17)

# cmake -DCMAKE_BUILD_TYPE=Release -DSC_PATH='/Users/spluta1/Documents/SC/SuperColliderBuild/SuperCollider' ..
# cmake --build . --config Release