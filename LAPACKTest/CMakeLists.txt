
file(GLOB SOURCES "*.cpp")
add_executable(LAPACKTest ${SOURCES})

if (WIN32)
    find_package(OpenBLAS CONFIG REQUIRED)
    target_link_libraries(LAPACKTest PRIVATE OpenBLAS::OpenBLAS)
    
    find_package(LAPACK CONFIG REQUIRED)
    target_link_libraries(LAPACKTest PRIVATE LAPACK::LAPACK)
elseif (UNIX)
    target_link_libraries(LAPACKTest PRIVATE lapack)
endif()

target_include_directories(LAPACKTest PRIVATE "${PROJECT_SOURCE_DIR}/utilities")