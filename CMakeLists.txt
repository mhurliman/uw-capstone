cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

project(uw-capstone VERSION 0.1.0)

if (WIN32)
    set(STATIC_LIB_EXT lib)
    set(SHARED_LIB_EXT dll)
elseif (UNIX)
    set(STATIC_LIB_EXT a)
    set(SHARED_LIB_EXT so)
endif()

function(link_3p_target_static target library)
    target_link_libraries(${target} PRIVATE ${PROJECT_SOURCE_DIR}/thirdparty/lib/lib${library}.${STATIC_LIB_EXT})
    target_include_directories(${target} PRIVATE ${PROJECT_SOURCE_DIR}/thirdparty/include/${library})
endfunction()

function(link_3p_target_shared target library)
    link_3p_target_static(${target} ${library})
    add_custom_command(TARGET ${target} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${PROJECT_SOURCE_DIR}/thirdparty/bin/lib${library}.${SHARED_LIB_EXT}
        $<TARGET_FILE_DIR:${target}>)
endfunction()

function(link_scalapack target)
    link_3p_target_static(${target} scalapack)
    target_compile_definitions(${target} PRIVATE 
        HAVE_LAPACK_CONFIG_H 
        LAPACK_COMPLEX_CPP
    )
    if (UNIX)
        target_link_libraries(${target} PRIVATE gfortran)
    endif()
endfunction()

function(link_openblas target)
if (WIN32)
    link_3p_target_shared(${target} openblas)
elseif (UNIX)
    link_3p_target_static(${target} openblas)
endif()
endfunction()

add_subdirectory(utilities)
add_subdirectory(qrfactor)
add_subdirectory(fft)
add_subdirectory(optests)
