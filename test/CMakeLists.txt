
file(GLOB SOURCES "*.cpp")
add_executable(test ${SOURCES})

target_compile_definitions(test PUBLIC -DHAVE_LAPACK_CONFIG_H -DLAPACK_COMPLEX_CPP)


target_link_libraries(test PRIVATE utilities) 