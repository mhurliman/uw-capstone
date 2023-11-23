
file(GLOB SOURCES "*.cpp")
add_executable(OpenBLASTest ${SOURCES})

if (WIN32)
    find_package(OpenBLAS CONFIG REQUIRED)
    target_link_libraries(OpenBLASTest PRIVATE OpenBLAS::OpenBLAS)
elseif(UNIX)
    # link openblas
    set(BLA_VENDER OpenBLAS)
    find_package(BLAS REQUIRED)
    if(BLAS_FOUND)
        include_directories(${BLAS_INCLUDE_DIRS})
        target_link_libraries(OpenBLASTest ${BLAS_LIBRARIES})
    endif(BLAS_FOUND)
endif()