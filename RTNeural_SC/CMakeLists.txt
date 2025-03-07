set(CMAKE_CXX_STANDARD 17)

cmake_minimum_required (VERSION 3.5)
project (RTNeural_SC)

# add_subdirectories manually
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../libsamplerate ${CMAKE_CURRENT_SOURCE_DIR}/../libsamplerate/build)

# set(RTNEURAL_STL ON ON CACHE BOOL "Use RTNeural with this backend" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../RTNeuralCPP ${CMAKE_CURRENT_SOURCE_DIR}/../RTNeuralCPP/build)

include_directories(${SC_PATH}/include/plugin_interface)
include_directories(${SC_PATH}/include/common)
# include_directories(${SC_PATH}/external_libraries/libsndfile/)

set(CMAKE_SHARED_MODULE_PREFIX "")
if(APPLE OR WIN32)
set(CMAKE_SHARED_MODULE_SUFFIX ".scx")
endif()

# add the plugin we are making
add_library(${PROJECT_NAME} MODULE RTNeuralUGen.cpp)

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

set(LIBSR_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../libsamplerate/build/src")
set(LIBSRI_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../libsamplerate/include")
include_directories(${LIBSR_PATH})
include_directories(${LIBSRI_PATH})

if(MSVC)
        target_link_libraries(${PROJECT_NAME} PUBLIC ${LIBSR_PATH}/Release/samplerate.lib)
else()
        target_link_libraries(${PROJECT_NAME} PUBLIC ${LIBSR_PATH}/libsamplerate.a)
endif()

# set installation directory
set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}")

message(STATUS "Install directory set to: ${CMAKE_INSTALL_PREFIX}")

install(DIRECTORY "${PROJECT_NAME}/Classes" "${PROJECT_NAME}/HelpSource" DESTINATION "${CMAKE_INSTALL_PREFIX}")

install(DIRECTORY DESTINATION "${CMAKE_INSTALL_PREFIX}/scx_files")

message(STATUS "CMake current binary directory: ${CMAKE_CURRENT_BINARY_DIR}")

if(APPLE)
  install(FILES ${CMAKE_BINARY_DIR}/${PROJECT_NAME}.scx DESTINATION ${CMAKE_INSTALL_PREFIX}/scx_files)
else()
  install(FILES ${CMAKE_BINARY_DIR}/Release/${PROJECT_NAME}.scx DESTINATION ${CMAKE_INSTALL_PREFIX}/scx_files)
endif()

# if(APPLE)
#   install(FILES ${CMAKE_BINARY_DIR}/${PROJECT_NAME}/${PROJECT_NAME}.scx DESTINATION ${CMAKE_INSTALL_PREFIX}/scx_files)
# else()
#   install(FILES ${CMAKE_BINARY_DIR}/${PROJECT_NAME}/${PROJECT_NAME}.scx DESTINATION ${CMAKE_INSTALL_PREFIX}/scx_files)
# endif()

