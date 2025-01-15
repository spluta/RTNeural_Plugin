include(${CMAKE_CURRENT_SOURCE_DIR}/../../max-sdk/source/max-sdk-base/script/max-pretarget.cmake)

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

message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")

if (UNIX)
	target_link_libraries(${PROJECT_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/../../build/libsamplerate/src/libsamplerate.a)
	# target_link_libraries(${PROJECT_NAME} /Users/spluta1/Dev/RTNeural_Plugin/RTNeural_Plugin/rtneural~_max/build/libsamplerate/src/libsamplerate.a)
	endif (UNIX)
if (WIN32)
    message ("Windows")
    target_link_libraries(${PROJECT_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/../../libsamplerate/lib/samplerate.lib)
endif (WIN32)

target_link_libraries(${PROJECT_NAME} RTNeural)

# add the plugin we are making
# add_library(${PROJECT_NAME} MODULE RTNeuralUGen.cpp)

include(${CMAKE_CURRENT_SOURCE_DIR}/../../max-sdk/source/max-sdk-base/script/max-posttarget.cmake)

# add_custom_command( 
# 	TARGET ${PROJECT_NAME} 
# 	POST_BUILD 
# 	COMMAND codesign --force --deep -s - "${CMAKE_CURRENT_SOURCE_DIR}/../../externals/rtneural.mxo"
# 	RESULT_VARIABLE result
# )