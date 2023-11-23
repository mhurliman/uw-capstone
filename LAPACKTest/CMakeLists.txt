
file(GLOB SOURCES "*.cpp")
add_executable(LAPACKTest ${SOURCES})

if (WIN32)
    find_package(LAPACK CONFIG REQUIRED)
    target_link_libraries(LAPACKTest PRIVATE LAPACK::LAPACK)
elseif (UNIX)
    target_link_libraries(LAPACKTest PRIVATE lapack)
endif()