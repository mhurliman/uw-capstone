# Platforms

Was able to get everything to compile on both Windows & Linux (Ubuntu). Using VS Code for code development and Windows Subsystem for Linux (WSL) to test the Linux build. Building using CMake on both platforms. Windows uses the VS2022 compiler (v17) and Linux utilizing g++.


# Package Dependencies

## Windows

Packages installed through the (vcpkg)[https://vcpkg.io/en/] package manager:
1. fftw3
2. mpi (required pre-installing [Microsoft MPI](https://www.microsoft.com/en-us/download/details.aspx?id=105289))
3. openblas
4. lapack-reference\[blas-select\]

Example - `.\vcpkg.exe install lapack-reference[blas-select]`

## Linux (Ubuntu)

Packages installed through the Debian package manager (apt-get):
1. libfftw3-dev
2. libopenmpi-dev
3. libopenblas-dev
4. liblapack-dev

Example - `sudo apt install libfftw3-dev`


# Build environment

## Windows
VS Code is a handy IDE when Visual Studio is inconvenient. Several extensions make development easier:
1. CMake Tools (auto-reconfigure on change, auto-build, debug/run kickoffs etc.)
2. C/C++ (Intellisense & debugging)
3. Python (auto-complete & debugging)

To get CMake working with C++ debugging it seems I had to change settings in the ~/.vscode/tasks.json file. Specifically I had to provide the path to the VsDevCmd.bat file.

```
{
    "windows": {
        "options": {
          "shell": {
            "executable": "cmd.exe",
            "args": [
              "/C",
              // The path to VsDevCmd.bat depends on the version of Visual Studio you have installed.
              "\"C:/Program Files/Microsoft Visual Studio/2022/Community/Common7/Tools/VsDevCmd.bat\"",
              "&&"
            ]
          }
        }
    },
    ...
}
```

To get CMake working with vcpkg I had to add a custom 'Kit' to target which specifies a CMake toolchain file. This is done by adding an option to ~/.vscode/cmake-kits.json (had to create it.)

```
[
    {
        "name": "MSVC 2022 Release - amd64 - vcpkg",
        "visualStudio": "f10c0764",
        "visualStudioArchitecture": "x64",
        "isTrusted": true,
        "preferredGenerator": {
          "name": "Visual Studio 17 2022",
          "platform": "x64",
          "toolset": "host=x64"
        },
        "toolchainFile": "$env{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
    }
]
```


## Linux

Exclusively using the command line to interact with CMake and test the build.
