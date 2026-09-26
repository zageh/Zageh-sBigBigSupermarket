cmake_minimum_required(VERSION 3.28)

if(NOT DEFINED CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER /usr/bin/g++-14 CACHE FILEPATH "C++ compiler" FORCE)
endif()

project(hello_world LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(main main.cpp)

target_sources(main
    PRIVATE
        FILE_SET CXX_MODULES FILES
            my_module.cpp
)
