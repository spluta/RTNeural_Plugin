include(${CMAKE_CURRENT_SOURCE_DIR}/../../max-sdk/source/max-sdk-base/script/max-pretarget.cmake)

set(CMAKE_OSX_DEPLOYMENT_TARGET "10.15" CACHE STRING "Minimum OS X deployment version" FORCE)

#############################################################
# MAX EXTERNAL
#############################################################

include_directories( 
	"${MAX_SDK_INCLUDES}"
	"${MAX_SDK_MSP_INCLUDES}"
	"${MAX_SDK_JIT_INCLUDES}"
)

file(GLOB PROJECT_SRC
     "*.h"
	 "*.c"
     "*.cpp"
)
add_library( 
	${PROJECT_NAME}
	MODULE
	${PROJECT_SRC}
)

set(LIBSR_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../libsamplerate/build/src")
set(LIBSRI_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../libsamplerate/include")
include_directories(${LIBSR_PATH})
include_directories(${LIBSRI_PATH})

if(WIN32)
        target_link_libraries(rtneural PUBLIC ${LIBSR_PATH}/Release/samplerate.lib)
else()
        target_link_libraries(rtneural PUBLIC ${LIBSR_PATH}/libsamplerate.a)
endif()

add_subdirectory(../../RTNeuralCPP RTNeuralCPP)
target_link_libraries(rtneural PUBLIC RTNeural)

message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")

include(${CMAKE_CURRENT_SOURCE_DIR}/../../max-sdk/source/max-sdk-base/script/max-posttarget.cmake)