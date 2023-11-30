
file(GLOB SOURCES "*.cpp")
add_executable(OpenBLASTest ${SOURCES})

link_openblas(OpenBLASTest)
