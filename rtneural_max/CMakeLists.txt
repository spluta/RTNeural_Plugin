cmake_minimum_required(VERSION 3.19)

set(MACOSX_DEPLOYMENT_TARGET 14.0)
set(CMAKE_CXX_STANDARD 17)

set(C74_BUILD_FAT YES)
set(CMAKE_BUILD_TYPE Release)

string(REGEX REPLACE "(.*)/" "" THIS_FOLDER_NAME "${CMAKE_CURRENT_SOURCE_DIR}")

project(${THIS_FOLDER_NAME})

set(C74_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/rtneural_max/externals")

message("CMAKE_OSX_ARCHITECTURES: ${CMAKE_OSX_ARCHITECTURES}")

if(APPLE)
  if (CMAKE_OSX_ARCHITECTURES STREQUAL "")
      execute_process(
        COMMAND uname -m
        OUTPUT_VARIABLE CURRENT_ARCH
        OUTPUT_STRIP_TRAILING_WHITESPACE
      )
      message("Current architecture is ${CURRENT_ARCH}")
      set(CMAKE_OSX_ARCHITECTURES ${CURRENT_ARCH})
  endif()
endif()


add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/source/rtneural~)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/source/rtneural)

add_subdirectory(../RTNeuralCPP RTNeuralCPP)

# set installation directory
set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/rtneural_max")

message(STATUS "Install directory set to: ${CMAKE_INSTALL_PREFIX}")

install(DIRECTORY "examples" DESTINATION "${CMAKE_INSTALL_PREFIX}")
