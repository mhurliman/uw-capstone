
file(GLOB SOURCES "*.cpp")
add_library(utilities ${SOURCES})

target_include_directories(utilities INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})