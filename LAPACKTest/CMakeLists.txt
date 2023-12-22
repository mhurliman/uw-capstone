
file(GLOB SOURCES "*.cpp")
add_executable(LAPACKTest ${SOURCES})

target_compile_definitions(LAPACKTest PUBLIC -DHAVE_LAPACK_CONFIG_H -DLAPACK_COMPLEX_CPP)

target_link_libraries(LAPACKTest PRIVATE scalapack)
target_link_libraries(LAPACKTest PRIVATE openblas)
target_link_libraries(LAPACKTest PRIVATE utilities) 