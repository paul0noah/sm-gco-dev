cmake_minimum_required(VERSION 3.16)
project(sm_gco)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_BUILD_TYPE Release)

## load external content
list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
# pybind11
if( BUILD_PYTHON_BINDINGS )
    include(pybind)
endif()
# Libigl
include(libigl)


# switch between python bindings and executable
if( BUILD_PYTHON_BINDINGS )
    message("Building Python Bindings")
    find_package(Python3 REQUIRED)
    include_directories(${PYTHON_INCLUDE_DIRS})
    pybind11_add_module(sm_gco pythonbinding/pythonwrappers.cpp)
    add_subdirectory(pythonbinding)
elseif( BUILD_MEX_FILE )
    message("Building MEX files")
    find_package(Matlab REQUIRED)
    set(PROJECT_NAME cpppy_myfunc)
    add_subdirectory(mex)
else()
    message("Building CPP Executable")
    add_executable(${PROJECT_NAME} main.cpp)
endif()


# set project include dir
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR})


# Add gco files
add_subdirectory(gco)

# Add your project files
add_subdirectory(src)


if( NOT BUILD_MEX_FILE )
    if(OpenMP_CXX_FOUND)
        message("Compiling with OPENMP")
        target_link_libraries(${PROJECT_NAME} PUBLIC igl::core OpenMP::OpenMP_CXX)
    else()
        message("Compiling without OPENMP")
        target_link_libraries(${PROJECT_NAME} PUBLIC igl::core)
    endif()
endif()

if(OpenMP_CXX_FOUND AND WITH_CUDA)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fopenmp -Xlinker -fgomp")
endif()
