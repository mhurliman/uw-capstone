
file(GLOB SOURCES "*.cpp")
add_executable(quad ${SOURCES})

target_compile_definitions(quad PUBLIC -DHAVE_LAPACK_CONFIG_H -DLAPACK_COMPLEX_CPP)

target_link_libraries(quad PRIVATE utilities) 