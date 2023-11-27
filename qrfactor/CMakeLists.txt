
file(GLOB SOURCES "*.cpp")
add_executable(qrfactor ${SOURCES})

target_link_libraries(qrfactor PUBLIC utilities)
target_include_directories(qrfactor PUBLIC "${PROJECT_BINARY_DIR}")

find_package(LAPACK REQUIRED)
target_link_libraries(qrfactor PUBLIC LAPACK::LAPACK)
