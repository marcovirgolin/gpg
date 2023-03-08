cmake_minimum_required(VERSION 3.17)  # CMake version check
cmake_policy(SET CMP0026 NEW)
cmake_policy(SET CMP0051 NEW)

# Variables
set(PROJECT_NAME gpg)
set(NUM_PRECISION 6)

set(CMAKE_CXX_FLAGS_VALGRIND
    "${CMAKE_CXX_FLAGS_VALGRIND} -g -O0 -DDEBUG -Wall -Wconversion -pedantic ${CXX_EXTRA}")
set(CMAKE_CXX_FLAGS_DEBUG
    "${CMAKE_CXX_FLAGS_DEBUG} -g -O0 -DDEBUG -Wall -Wconversion ${CXX_EXTRA}")
set(CMAKE_CXX_FLAGS_RELEASE
    "${CMAKE_CXX_FLAGS_RELEASE} -O2 -DNDEBUG ${CXX_EXTRA}")

# Create project
project(${PROJECT_NAME})               
# Enable c++17 standard
set(CMAKE_CXX_STANDARD 17)  


### Inclusions

# Include Eigen
find_package(Eigen3 REQUIRED)
set(EIGEN3_INCLUDE_DIR "${EIGEN3_INCLUDE_DIR}")
if(NOT EIGEN3_INCLUDE_DIR)
  message( FATAL_ERROR "Please point the environment variable EIGEN3_INCLUDE_DIR to the include directory of your Eigen3 installation.")
endif()

# Include SWIG for libraries
find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})

# Include python libraries
find_package(Python3 COMPONENTS NumPy REQUIRED)
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${PYTHON_INCLUDE_PATH})


### Compilation flags

# Set correct compilation flags
if(CMAKE_BUILD_TYPE STREQUAL "debug")
  set(ALL_COMPILE_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")
elseif(CMAKE_BUILD_TYPE STREQUAL "release")
  set(ALL_COMPILE_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
elseif(CMAKE_BUILD_TYPE STREQUAL "valgrind")
  set(ALL_COMPILE_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_VALGRIND}")
endif()


# Add executable target with source files listed in SOURCE_FILES variable
add_executable(${PROJECT_NAME} src/main.cpp )

# Seek directories for compilation
target_include_directories(${PROJECT_NAME} PRIVATE src)
target_include_directories(${PROJECT_NAME} PUBLIC ${EIGEN3_INCLUDE_DIR})
# linking
target_link_libraries(${PROJECT_NAME} ${PYTHON_LIBRARIES} Python3::NumPy)

# Do not check for CTRL+C from Python for the standalone executable
target_compile_definitions(${PROJECT_NAME} PUBLIC CLI=1)
target_compile_definitions(${PROJECT_NAME} PUBLIC NUM_PRECISION=${NUM_PRECISION})

# SWIG
set_property(SOURCE src/swig/python/pyface.i PROPERTY CPLUSPLUS ON)
swig_add_library(pyface
  TYPE MODULE
  LANGUAGE python
  SOURCES src/swig/python/pyface.i src/ src/swig/python/pyface.hpp ${EIGEN3_INCLUDE_DIR}
)

#target_include_directories(${PROJECT_NAME} PUBLIC "include/")
set_property(TARGET pyface PROPERTY SWIG_COMPILE_OPTIONS -c++)
target_include_directories(pyface PRIVATE src)
target_include_directories(pyface PRIVATE src/swig/python)
target_include_directories(pyface PUBLIC ${EIGEN3_INCLUDE_DIR})
target_link_libraries(pyface ${PYTHON_LIBRARIES} Python3::NumPy)
target_compile_definitions(pyface PUBLIC PYLIB=1)
target_compile_definitions(pyface PUBLIC NUM_PRECISION=${NUM_PRECISION})
