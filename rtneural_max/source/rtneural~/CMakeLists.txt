include(${CMAKE_CURRENT_SOURCE_DIR}/../../max-sdk/source/max-sdk-base/script/max-pretarget.cmake)
# include(${CMAKE_CURRENT_SOURCE_DIR}/../../max-pretarget.cmake)

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

include(${CMAKE_CURRENT_SOURCE_DIR}/../../max-sdk/source/max-sdk-base/script/max-posttarget.cmake)

