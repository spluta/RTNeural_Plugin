cmake_minimum_required(VERSION 3.19)

set(CMAKE_CXX_STANDARD 17)
set(C74_BUILD_FAT YES)
set(CMAKE_BUILD_TYPE Release)

string(REGEX REPLACE "(.*)/" "" THIS_FOLDER_NAME "${CMAKE_CURRENT_SOURCE_DIR}")
project(${THIS_FOLDER_NAME})

# if (APPLE)
#     if (${CMAKE_GENERATOR} MATCHES "Xcode")
#             if (${XCODE_VERSION} VERSION_LESS 10)
#                 message(STATUS "Xcode 10 or higher is required. Please install from the Mac App Store.")
#                 return ()
#             elseif(${XCODE_VERSION} VERSION_GREATER_EQUAL 12)
#                 set(C74_BUILD_FAT YES)
#             endif ()
#     endif ()

#     if (NOT CMAKE_OSX_ARCHITECTURES)
#         if(C74_BUILD_FAT)
#             set(CMAKE_OSX_ARCHITECTURES "x86_64;arm64" CACHE STRING "macOS architecture" FORCE)
#         else()
#             set(CMAKE_OSX_ARCHITECTURES ${CMAKE_SYSTEM_PROCESSOR} CACHE STRING "macOS architecture" FORCE)
#         endif()
#         message("CMAKE_OSX_ARCHITECTURES set to ${CMAKE_OSX_ARCHITECTURES}")
#     endif()
# endif()

set(C74_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/externals")

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

# add the submodule
add_subdirectory(../RTNeuralCPP ../build/RTNeuralCPP)

if (UNIX)
    message ("LINUX or MAC OS X")
    set(BUILD_TESTING OFF CACHE BOOL "Don't build libsamplerate tests!")
    add_subdirectory(../libsamplerate ../build/libsamplerate)
    message(STATUS "VERSION for libsamplerate: ${libsamplerate_version}")
    # target_link_libraries(rtneural ${CMAKE_CURRENT_SOURCE_DIR}/../libsamplerate/build/src/libsamplerate.a)
endif (UNIX)
if (WIN32)
    message ("Windows")
    # target_link_libraries(rtneural ${CMAKE_CURRENT_SOURCE_DIR}/../libsamplerate/lib/samplerate.lib)
endif (WIN32)

# target_link_libraries(rtneural RTNeural)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/source/rtneural~)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/source/rtneural)

# run codesign on the output files
# file(GLOB EXTERNAL_FILES "${C74_LIBRARY_OUTPUT_DIRECTORY}/*.mxo")
# foreach(FILE ${EXTERNAL_FILES})
#   execute_process(COMMAND codesign --force --deep -s - "${FILE}"
#           RESULT_VARIABLE result)
#   if(result)
#     message(FATAL_ERROR "Failed to codesign ${FILE}")
#   else()
#     message(STATUS "Successfully codesigned ${FILE}")
#   endif()
# endforeach()

# if (APPLE AND NOT "${PROJECT_NAME}" MATCHES "_test")
# 	add_custom_command( 
# 		TARGET ${PROJECT_NAME} 
# 		POST_BUILD 
# 		COMMAND cp "${CMAKE_CURRENT_LIST_DIR}/PkgInfo" "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${${PROJECT_NAME}_EXTERN_OUTPUT_NAME}.mxo/Contents/PkgInfo" 
# 		COMMENT "Copy PkgInfo" 
# 	)
# endif ()

