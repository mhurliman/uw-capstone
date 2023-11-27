
file(GLOB SOURCES "*.cpp")
add_executable(fft ${SOURCES})

if (WIN32)
find_package(FFTW3 CONFIG REQUIRED)
target_link_libraries(fft PRIVATE FFTW3::fftw3)
elseif (UNIX)
target_link_libraries(fft PRIVATE fftw3)
endif()