
file(GLOB SOURCES "*.cpp")
add_executable(OpenBLASTest ${SOURCES})

target_link_libraries(OpenBLASTest PRIVATE openblas)
