cmake_minimum_required(VERSION 3.18)
project(sort_tracker CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)   # needed later for the pybind11 module

# ---- Eigen (header-only matrix library), searched in this order:
#   1) third_party/eigen/  (Eigen unzipped by hand next to this file)
#   2) an Eigen installed on the system
#   3) automatic download
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen/Eigen/Dense")
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    set_target_properties(Eigen3::Eigen PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen")
else()
    find_package(Eigen3 3.3 QUIET NO_MODULE)
endif()

if(NOT TARGET Eigen3::Eigen)
    message(STATUS "Eigen not found: downloading it")
    include(FetchContent)
    FetchContent_Declare(
        eigen
        URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
        SOURCE_SUBDIR do_not_build    # only the headers are needed
    )
    FetchContent_MakeAvailable(eigen)
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    set_target_properties(Eigen3::Eigen PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${eigen_SOURCE_DIR}")
endif()

# ---- SORT library
add_library(sort_core STATIC
    src/iou.cpp
    src/kalman_box_tracker.cpp
    src/matching.cpp
    src/sort.cpp
)
target_include_directories(sort_core PUBLIC include)
target_link_libraries(sort_core PUBLIC Eigen3::Eigen)

# ---- Tests
enable_testing()
foreach(name test_iou test_kalman test_sort)
    add_executable(${name} tests/${name}.cpp)
    target_link_libraries(${name} PRIVATE sort_core)
    add_test(NAME ${name} COMMAND ${name})
endforeach()

# ---- Python module (pybind11)
find_package(Python 3 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(sort_tracker src/bindings.cpp)
target_link_libraries(sort_tracker PRIVATE sort_core)
