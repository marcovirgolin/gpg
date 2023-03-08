cmake_minimum_required(VERSION 3.17)  # CMake version check
cmake_policy(SET CMP0026 NEW)
cmake_policy(SET CMP0051 NEW)

# Variables
set(PROJECT_NAME gpg)
set(PY_LIB_NAME _pb_gpg)
set(NUM_PRECISION 6)

set(CMAKE_CXX_FLAGS_VALGRIND
    "${CMAKE_CXX_FLAGS_VALGRIND} -g -O0 -DDEBUG -Wall -Wconversion -pedantic ${CXX_EXTRA}")
set(CMAKE_CXX_FLAGS_DEBUG
    "${CMAKE_CXX_FLAGS_DEBUG} -g -O0 -DDEBUG -Wall -Wconversion ${CXX_EXTRA}")
set(CMAKE_CXX_FLAGS_RELEASE
    "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG ${CXX_EXTRA}")

# Create project
project(${PROJECT_NAME})               
# Enable c++17 standard
set(CMAKE_CXX_STANDARD 17)  
set(PYBIND11_CPP_STANDARD -std=c++17)

### Inclusions

# Include Eigen
find_package(Eigen3 REQUIRED)
set(EIGEN3_INCLUDE_DIR "${EIGEN3_INCLUDE_DIR}")
if(NOT EIGEN3_INCLUDE_DIR)
  message( FATAL_ERROR "Please point the environment variable EIGEN3_INCLUDE_DIR to the include directory of your Eigen3 installation.")
endif()

# Include python libraries
find_package(Python3 COMPONENTS NumPy REQUIRED)
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${PYTHON_INCLUDE_PATH})

# Include pybind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)


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
target_link_libraries(${PROJECT_NAME} ${PYTHON_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${PYTHON_LIBRARIES} Python3::NumPy)
target_link_libraries(${PROJECT_NAME} pybind11::headers)

# Do not check for CTRL+C from Python for the standalone executable
target_compile_definitions(${PROJECT_NAME} PUBLIC CLI=1)
target_compile_definitions(${PROJECT_NAME} PUBLIC NUM_PRECISION=${NUM_PRECISION})

### Pybind interface (pyfe)
#pybind11_add_module(pyfe src/python_interface.hpp)
add_library(${PY_LIB_NAME} MODULE src/python_interface.cpp)
set_target_properties(${PY_LIB_NAME} PROPERTIES LINKER_LANGUAGE CXX)
target_compile_definitions(${PY_LIB_NAME} PUBLIC NUM_PRECISION=${NUM_PRECISION})
target_include_directories(${PY_LIB_NAME} PUBLIC ${EIGEN3_INCLUDE_DIR})
target_link_libraries(${PY_LIB_NAME} PRIVATE pybind11::pybind11 pybind11::headers pybind11::module pybind11::lto Python3::NumPy)
pybind11_extension(${PY_LIB_NAME})
if(NOT MSVC AND NOT ${CMAKE_BUILD_TYPE} MATCHES Debug|RelWithDebInfo)
    # Strip unnecessary sections of the binary on Linux/macOS
    pybind11_strip(${PY_LIB_NAME})
endif()
