cmake_minimum_required(VERSION 3.5)
project(inflo)

set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(CMAKE_CXX_FLAGS "-O3 -std=gnu++11 -fopenmp -fexceptions")

include_directories(
  include
)

add_executable(inflo 
      src/main.cpp
      src/ParameterManager.cpp
    )
find_package(PkgConfig REQUIRED)
find_package(PCL REQUIRED)
find_package(VTK REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
target_link_libraries(inflo ${PCL_LIBRARIES})
include_directories(${VTK_INCLUDE_DIRS})
target_link_libraries(inflo ${VTK_LIBRARIES})

