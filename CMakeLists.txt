# cmake_minimum_required(VERSION 3.19)
# set(lib_name "RTNeural_Plugin")
# project("${lib_name}")

# if (NOT MSVC)
#   # specify the C++ standard
#   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17" CACHE STRING "c++ version" FORCE)
#   if(CMAKE_COMPILER_IS_CLANG)
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
#   endif()
# endif()


# # set some options for libsamplerate

# option(LIBSAMPLERATE_EXAMPLES "libsamplerate: build examples" OFF) 
# option(BUILD_TESTING "libsamplerate: run testing" OFF)

# # add_subdirectories manually
# add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/libsamplerate ${CMAKE_CURRENT_SOURCE_DIR}/libsamplerate/build)

# # set(RTNEURAL_STL ON ON CACHE BOOL "Use RTNeural with this backend" FORCE)
# add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/RTNeuralCPP ${CMAKE_CURRENT_SOURCE_DIR}/RTNeuralCPP/build)

# add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/rtneural_max)
# add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/RTNeural_SC)
# add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/rtneural_pd)


