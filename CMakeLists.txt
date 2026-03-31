cmake_minimum_required(VERSION 3.18)

function(debug_message messages)
  message(STATUS "\n")
  message(STATUS "${messages}")
  message(STATUS "\n")
endfunction()

# Variables
set(PROJECT_NAME gpg)
set(PY_LIB_NAME _pb_gpg)
set(NUM_PRECISION 6)

project(${PROJECT_NAME} LANGUAGES CXX)

set(CMAKE_CXX_FLAGS_VALGRIND
    "${CMAKE_CXX_FLAGS_VALGRIND} -g -O0 -DDEBUG -Wall -Wconversion -pedantic ${CXX_EXTRA}")
set(CMAKE_CXX_FLAGS_DEBUG
    "${CMAKE_CXX_FLAGS_DEBUG} -g -O0 -DDEBUG -Wall -Wconversion ${CXX_EXTRA}")
set(CMAKE_CXX_FLAGS_RELEASE
    "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG ${CXX_EXTRA}")

# Enable c++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

### Inclusions

# Include Eigen
find_package(Eigen3 REQUIRED)

# Include python libraries
find_package(Python COMPONENTS Interpreter Development.Module NumPy REQUIRED)
debug_message("Found Python\n\tDIRS: ${Python_INCLUDE_DIRS}\t${Python_NumPy_INCLUDE_DIRS}\n\tLIBS: ${Python_LIBRARIES}")

# Include pybind11
find_package(pybind11 CONFIG REQUIRED)


# Add executable target with source files listed in SOURCE_FILES variable
add_executable(${PROJECT_NAME} src/main.cpp )

# Seek directories for compilation
target_include_directories(${PROJECT_NAME} PRIVATE src)
# linking
target_link_libraries(${PROJECT_NAME} PRIVATE Eigen3::Eigen Python::NumPy pybind11::headers)

# Do not check for CTRL+C from Python for the standalone executable
target_compile_definitions(${PROJECT_NAME} PUBLIC CLI=1)
target_compile_definitions(${PROJECT_NAME} PUBLIC NUM_PRECISION=${NUM_PRECISION})

### Pybind interface (pyfe)
pybind11_add_module(${PY_LIB_NAME} MODULE src/python_interface.cpp)
target_compile_definitions(${PY_LIB_NAME} PUBLIC NUM_PRECISION=${NUM_PRECISION})
target_include_directories(${PY_LIB_NAME} PRIVATE src)
target_link_libraries(${PY_LIB_NAME} PRIVATE Eigen3::Eigen Python::NumPy)
if(NOT MSVC AND NOT ${CMAKE_BUILD_TYPE} MATCHES Debug|RelWithDebInfo)
    # Strip unnecessary sections of the binary on Linux/macOS
    pybind11_strip(${PY_LIB_NAME})
endif()

install(DIRECTORY src/pypkg/pygpg DESTINATION . FILES_MATCHING PATTERN "*.py")
install(TARGETS ${PY_LIB_NAME}
        LIBRARY DESTINATION pygpg
        RUNTIME DESTINATION pygpg
        ARCHIVE DESTINATION pygpg)
