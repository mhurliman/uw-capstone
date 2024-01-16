
file(GLOB SOURCES "*.cpp")
add_executable(optests ${SOURCES})

target_compile_definitions(optests PUBLIC -DHAVE_LAPACK_CONFIG_H -DLAPACK_COMPLEX_CPP)


target_link_libraries(optests PRIVATE utilities) 