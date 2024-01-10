
file(GLOB SOURCES "*.cpp")
add_library(utilities ${SOURCES})

target_include_directories(utilities INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(utilities PRIVATE scalapack)
target_link_libraries(utilities PRIVATE openblas)
target_link_libraries(utilities PRIVATE mpi)
target_link_libraries(utilities PRIVATE gfortran)