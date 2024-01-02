
file(GLOB SOURCES "*.cpp")
add_executable(pdgemm_test ${SOURCES})

target_compile_definitions(pdgemm_test PUBLIC -DHAVE_LAPACK_CONFIG_H -DLAPACK_COMPLEX_CPP)


target_link_libraries(pdgemm_test PRIVATE scalapack)
target_link_libraries(pdgemm_test PRIVATE openblas)
target_link_libraries(pdgemm_test PRIVATE mpi)
target_link_libraries(pdgemm_test PRIVATE gfortran)
target_link_libraries(pdgemm_test PRIVATE utilities) 