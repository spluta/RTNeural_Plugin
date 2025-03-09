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

install(DIRECTORY "examples" DESTINATION "${CMAKE_INSTALL_PREFIX}")

message(STATUS "Install directory set to: ${CMAKE_INSTALL_PREFIX}")

set(ORIGINAL_FILE_NAME "${CMAKE_CURRENT_BINARY_DIR}/rtneural")






# # set minimal version of cmake
# cmake_minimum_required(VERSION 3.18) 

# set(CMAKE_CXX_STANDARD 17)

# project(rt_neural_tilde) 

# set(PD_SOURCES_PATH ${PD_PATH})
# set(PD_EXTENSION "so")

# include(pd.cmake)
# set(CMAKE_BUILD_TYPE Release)

# pd_add_external(rtneural~ rtneural~.cpp TARGET rtneural_tilde)
# pd_add_external(rtneural rtneural.cpp TARGET rtneural)

# add_subdirectory(../RTNeuralCPP ../RTNeuralCPP)
# target_link_libraries(rtneural_tilde optimized RTNeural)
# target_link_libraries(rtneural optimized RTNeural)

# if (UNIX)
#         message ("LINUX or MAC OS X")
#         set(BUILD_TESTING OFF CACHE BOOL "Don't build libsamplerate tests!")
#         add_subdirectory(../libsamplerate libsamplerate)
#         message(STATUS "VERSION for libsamplerate: ${libsamplerate_version}")
#         target_link_libraries(rtneural_tilde ${CMAKE_CURRENT_SOURCE_DIR}/../libsamplerate/build/src/libsamplerate.a)
#         target_link_libraries(rtneural ${CMAKE_CURRENT_SOURCE_DIR}/../libsamplerate/build/src/libsamplerate.a)
# endif (UNIX)
# if (WIN32)
#         message ("Windows")
#         target_link_libraries(rtneural_tilde ${CMAKE_CURRENT_SOURCE_DIR}/../libsamplerate/lib/samplerate.lib)
#         target_link_libraries(rtneural ${CMAKE_CURRENT_SOURCE_DIR}/../libsamplerate/lib/samplerate.lib)
# endif (WIN32)

# set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/rtneural_pd")

# message(STATUS "Install directory set to: ${CMAKE_INSTALL_PREFIX}")

# install(DIRECTORY "examples" DESTINATION "${CMAKE_INSTALL_PREFIX}")

# install(FILES "${CMAKE_CURRENT_BINARY_DIR}/build/rtneural${PD_EXTENSION}" DESTINATION ${CMAKE_INSTALL_PREFIX})
# install(FILES "${CMAKE_CURRENT_BINARY_DIR}/rtneural~${PD_EXTENSION}" DESTINATION ${CMAKE_INSTALL_PREFIX})

