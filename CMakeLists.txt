cmake_minimum_required(VERSION 3.10)
project(holo)

# Set C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(JAVA_HOME "/usr/lib/jvm/java-23-openjdk")
set(JAVA_AWT_LIBRARY "${JAVA_HOME}/lib/libjawt.so")
set(JAVA_JVM_LIBRARY "${JAVA_HOME}/lib/server/libjvm.so")
set(JAVA_INCLUDE_PATH "${JAVA_HOME}/include")
set(JAVA_INCLUDE_PATH2 "${JAVA_HOME}/include/linux")
set(JAVA_AWT_INCLUDE_PATH "${JAVA_HOME}/include")

# Boost configuration
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)

# Add our custom modules paths
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

set(VTK_WRAP_JAVA OFF CACHE BOOL "Turn off VTK Java wrapping")
set(VTK_JAVA_INSTALL OFF CACHE BOOL "Turn off VTK Java installation")

# Fetch CLI11
include(FetchContent)
FetchContent_Declare(
        CLI11
        GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
        GIT_TAG        v2.1.2
)
FetchContent_MakeAvailable(CLI11)

# Workaround: set CLI11_DIR so that VTK can find it.
if(NOT CLI11_DIR)
    set(CLI11_DIR ${cli11_SOURCE_DIR})
endif()
list(APPEND CMAKE_PREFIX_PATH ${CLI11_DIR})

# Find required packages
find_package(Boost 1.86.0 REQUIRED COMPONENTS system thread iostreams program_options)
find_package(OpenCV 4.11.0 REQUIRED)
find_package(GSL REQUIRED)
find_package(GTest REQUIRED)

# Disable VTK GDAL support if not needed.
set(VTK_USE_GDAL OFF CACHE BOOL "Disable GDAL support in VTK")

find_package(VTK REQUIRED)
if(VTK_FOUND)
    include(${VTK_USE_FILE})
endif()

# Include directories
set(INC_OWN "${CMAKE_SOURCE_DIR}/include")
include_directories(${INC_OWN})
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${GSL_INCLUDE_DIRS})
include_directories(${GTEST_INCLUDE_DIRS})
include_directories(${cli11_SOURCE_DIR}/include)

# Set additional compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3 -Wreturn-type -D__extern_always_inline=inline")

# Create an extra variable for Boost.Program_options
if(Boost_PROGRAM_OPTIONS_LIBRARY)
    message(STATUS "Using Boost.Program_options library: ${Boost_PROGRAM_OPTIONS_LIBRARY}")
    set(EXTRA_BOOST_LIBS ${Boost_PROGRAM_OPTIONS_LIBRARY})
else()
    message(STATUS "Boost_PROGRAM_OPTIONS_LIBRARY not set; assuming it is in Boost_LIBRARIES")
    set(EXTRA_BOOST_LIBS "")
endif()

# --------------------------------------------------------------------
# Create a library target for your core code.
# Exclude main application files like main.cpp.
# --------------------------------------------------------------------
add_library(holo_core
        src/reference.cpp
        src/fft.cpp
        src/filters.cpp
        src/gerch.cpp
        src/optimize.cpp
        src/global.cpp
)

target_link_libraries(holo_core
        ${Boost_LIBRARIES}
        ${EXTRA_BOOST_LIBS}
        ${OpenCV_LIBS}
        ${GSL_LIBRARIES}
        ${VTK_LIBRARIES}
)

# Main executables
add_executable(holo main.cpp)
add_executable(holorunner src/holorunner.cpp)

target_link_libraries(holo holo_core)
target_link_libraries(holorunner holo_core)

# --------------------------------------------------------------------
# Google Test integration: add test executable
# Include both test sources and core sources in the test target.
# --------------------------------------------------------------------
file(GLOB TEST_SOURCES "test/*.cpp")
if(TEST_SOURCES)
    add_executable(runTests
            ${TEST_SOURCES}
            test/test_main.cpp
            # No need to explicitly list each test file if they match the glob,
            # but if you want, you can list them.
    )
    target_link_libraries(runTests
            holo_core
            ${GTEST_LIBRARIES}
            ${GTEST_MAIN_LIBRARIES}
            ${Boost_LIBRARIES}
            ${EXTRA_BOOST_LIBS}
            ${OpenCV_LIBS}
            ${VTK_LIBRARIES}
    )
    enable_testing()
    add_test(NAME runTests COMMAND runTests)
endif()
