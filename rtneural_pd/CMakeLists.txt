# set minimal version of cmake
cmake_minimum_required(VERSION 3.18) 

set(CMAKE_CXX_STANDARD 17)

project(rt_neural_tilde) 

set(PD_SOURCES_PATH ${PD_PATH})
include(pd.cmake)
set(CMAKE_BUILD_TYPE Release)

pd_add_external(rtneural~ rtneural~.cpp TARGET rtneural_tilde)
pd_add_external(rtneural rtneural.cpp TARGET rtneural)

add_subdirectory(../RTNeuralCPP ../RTNeuralCPP)
target_link_libraries(rtneural_tilde optimized RTNeural)
target_link_libraries(rtneural optimized RTNeural)

if (UNIX)
        message ("LINUX or MAC OS X")
        set(BUILD_TESTING OFF CACHE BOOL "Don't build libsamplerate tests!")
        add_subdirectory(../libsamplerate ../libsamplerate)
        message(STATUS "VERSION for libsamplerate: ${libsamplerate_version}")
        target_link_libraries(rtneural_tilde ${CMAKE_CURRENT_SOURCE_DIR}/../libsamplerate/build/src/libsamplerate.a)
        target_link_libraries(rtneural ${CMAKE_CURRENT_SOURCE_DIR}/../libsamplerate/build/src/libsamplerate.a)
endif (UNIX)
if (WIN32)
        message ("Windows")
        target_link_libraries(rtneural_tilde ${CMAKE_CURRENT_SOURCE_DIR}/../libsamplerate/lib/samplerate.lib)
        target_link_libraries(rtneural ${CMAKE_CURRENT_SOURCE_DIR}/../libsamplerate/lib/samplerate.lib)
endif (WIN32)
