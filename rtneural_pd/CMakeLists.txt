# set minimal version of cmake
cmake_minimum_required(VERSION 3.5) 

set(CMAKE_CXX_STANDARD 17)

project(rt_neural_tilde) 

set(PD_SOURCES_PATH ${PD_PATH})

message(STATUS "PD_PATH is set to: ${PD_PATH}")
message(STATUS "PD_SOURCES_PATH is set to: ${PD_SOURCES_PATH}")

set(PD_EXTENSION "so")

include(pd.cmake)
set(CMAKE_BUILD_TYPE Release)

pd_add_external(rtneural~ rtneural~.cpp TARGET rtneural_tilde)
pd_add_external(rtneural rtneural.cpp TARGET rtneural)

set(LIBSR_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../libsamplerate/build/src")
set(LIBSRI_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../libsamplerate/include")
include_directories(${LIBSR_PATH})
include_directories(${LIBSRI_PATH})

if(MSVC)
        target_link_libraries(rtneural_tilde PUBLIC ${LIBSR_PATH}/samplerate.lib)
        target_link_libraries(rtneural PUBLIC ${LIBSR_PATH}/samplerate.lib)
else()
        target_link_libraries(rtneural_tilde PUBLIC ${LIBSR_PATH}/libsamplerate.a)
        target_link_libraries(rtneural PUBLIC ${LIBSR_PATH}/libsamplerate.a)
endif()

# set installation directory
set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/rtneural_pd")

message(STATUS "Install directory set to: ${CMAKE_INSTALL_PREFIX}")

install(DIRECTORY "examples" DESTINATION "${CMAKE_INSTALL_PREFIX}")

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/rtneural.so" DESTINATION ${CMAKE_INSTALL_PREFIX})
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/rtneural~.so" DESTINATION ${CMAKE_INSTALL_PREFIX})


