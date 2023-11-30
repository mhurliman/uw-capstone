
file(GLOB SOURCES "*.cpp")
add_executable(LAPACKTest ${SOURCES})

link_openblas(LAPACKTest)
link_scalapack(LAPACKTest)
target_link_libraries(LAPACKTest PRIVATE utilities)