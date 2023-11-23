project(OpenMPITest)

file(GLOB SOURCES "*.cpp")
add_executable(OpenMPITest ${SOURCES})

find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
target_link_libraries(OpenMPITest ${MPI_C_LIBRARIES})
