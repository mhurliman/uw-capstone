
file(GLOB SOURCES "*.cpp")
add_executable(qho ${SOURCES})

target_compile_definitions(qho PUBLIC -DHAVE_LAPACK_CONFIG_H -DLAPACK_COMPLEX_CPP)


target_link_libraries(qho PRIVATE utilities) 