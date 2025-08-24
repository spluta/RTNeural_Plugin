cmake_minimum_required(VERSION 3.5) 

set(CMAKE_CXX_STANDARD 17)

project(rt_neural_tilde) 

set(PD_SOURCES_PATH ${PD_PATH})

message(STATUS "PD_PATH is set to: ${PD_PATH}")
message(STATUS "PD_SOURCES_PATH is set to: ${PD_SOURCES_PATH}")

if (WIN32)
        set (PDBINDIR "${CMAKE_CURRENT_SOURCE_DIR}/bin55-2")
endif()

include(pd.cmake)
set(CMAKE_BUILD_TYPE Release)

pd_add_external(rtneural rtneural.cpp TARGET rtneuralpd)
pd_add_external(rtneural~ rtneural~.cpp TARGET rtneuralpd_tilde)
pd_add_external(softmax softmax.cpp TARGET softmaxpd)

add_subdirectory(../libsamplerate libsamplerate)
add_subdirectory(../RTNeuralCPP RTNeuralCPP)

target_link_libraries(rtneuralpd_tilde LINK_PUBLIC RTNeural)
target_link_libraries(rtneuralpd LINK_PUBLIC RTNeural)

option(LIBSAMPLERATE_EXAMPLES "libsamplerate: build examples" OFF) 
option(BUILD_TESTING "libsamplerate: run testing" OFF)

set(LIBSR_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../libsamplerate/build/src")
set(LIBSRI_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../libsamplerate/include")
include_directories(${LIBSR_PATH})
include_directories(${LIBSRI_PATH})

if(MSVC)
        target_link_libraries(rtneuralpd_tilde PUBLIC ${LIBSR_PATH}/Release/samplerate.lib)
        target_link_libraries(rtneuralpd PUBLIC ${LIBSR_PATH}/Release/samplerate.lib)
else()
        target_link_libraries(rtneuralpd_tilde PUBLIC ${LIBSR_PATH}/libsamplerate.a)
        target_link_libraries(rtneuralpd PUBLIC ${LIBSR_PATH}/libsamplerate.a)
endif()


# set installation directory
set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/rtneural_pd")

message(STATUS "Install directory set to: ${CMAKE_INSTALL_PREFIX}")

install(DIRECTORY "rtneural_examples/" DESTINATION "${CMAKE_INSTALL_PREFIX}")

message(STATUS "Install directory set to: ${CMAKE_INSTALL_PREFIX}")

set(ORIGINAL_FILE_NAME "${CMAKE_CURRENT_BINARY_DIR}/rtneural")
