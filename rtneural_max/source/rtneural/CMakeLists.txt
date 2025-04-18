include(${CMAKE_CURRENT_SOURCE_DIR}/../../max-sdk/source/max-sdk-base/script/max-pretarget.cmake)


if(APPLE)
	execute_process(
		COMMAND sw_vers -productVersion
		OUTPUT_VARIABLE MACOS_VERSION
		OUTPUT_STRIP_TRAILING_WHITESPACE
	)
	string(REGEX REPLACE "([0-9]+\\.[0-9]+)\\..*" "\\1" MACOS_VERSION ${MACOS_VERSION})
	set(CMAKE_OSX_DEPLOYMENT_TARGET "${MACOS_VERSION}" CACHE STRING "Minimum OS X deployment version" FORCE)
endif()


# set(CMAKE_OSX_DEPLOYMENT_TARGET "11" CACHE STRING "Minimum OS X deployment version" FORCE)

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

if(MSVC)
        target_link_libraries(rtneural PUBLIC ${LIBSR_PATH}/Release/samplerate.lib)
else()
        target_link_libraries(rtneural PUBLIC ${LIBSR_PATH}/libsamplerate.a)
endif()

target_link_libraries(rtneural PUBLIC RTNeural)

message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")

include(${CMAKE_CURRENT_SOURCE_DIR}/../../max-sdk/source/max-sdk-base/script/max-posttarget.cmake)
