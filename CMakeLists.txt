cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

project(uw-capstone VERSION 0.1.0)

add_subdirectory(utilities)
add_subdirectory(qrfactor)
add_subdirectory(OpenBLASTest)

if (UNIX)
    add_subdirectory(OpenMPITest)
endif()
