cmake_minimum_required(VERSION 3.10)
project(holo)

# Set C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Java settings (if needed)
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

# Add custom module paths
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

# VTK settings
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

# Find required packages
find_package(Boost 1.55.0 REQUIRED COMPONENTS system thread iostreams program_options)
find_package(OpenCV 4.11.0 REQUIRED)
find_package(GSL REQUIRED)
find_package(GTest REQUIRED)
find_package(VTK REQUIRED)

if(VTK_FOUND)
    include(${VTK_USE_FILE})
endif()

# Set include directories
set(INC_OWN "${CMAKE_SOURCE_DIR}/include")
include_directories(${INC_OWN})
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${GSL_INCLUDE_DIRS})
include_directories(${GTEST_INCLUDE_DIRS})
include_directories(${cli11_SOURCE_DIR}/include)

# Set additional compiler flags (removing invalid ones)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3 -Wreturn-type -D__extern_always_inline=inline")

# Create an extra variable for Boost.Program_options
if(Boost_PROGRAM_OPTIONS_LIBRARY)
    message(STATUS "Using Boost.Program_options library: ${Boost_PROGRAM_OPTIONS_LIBRARY}")
    set(EXTRA_BOOST_LIBS ${Boost_PROGRAM_OPTIONS_LIBRARY})
else()
    message(STATUS "Boost_PROGRAM_OPTIONS_LIBRARY not set; assuming it is in Boost_LIBRARIES")
    set(EXTRA_BOOST_LIBS "")
endif()

# Main executables
add_executable(holo
        main.cpp
        src/reference.cpp
        src/fft.cpp
        src/filters.cpp
        src/gerch.cpp
        include/optimize.hpp
        src/global.cpp
)

add_executable(holorunner
        src/holorunner.cpp
        src/reference.cpp
        src/fft.cpp
        src/filters.cpp
        src/global.cpp
)

target_link_libraries(holo
        ${Boost_LIBRARIES}
        ${EXTRA_BOOST_LIBS}
        ${OpenCV_LIBS}
        ${GSL_LIBRARIES}
        ${VTK_LIBRARIES}
)

target_link_libraries(holorunner
        ${Boost_LIBRARIES}
        ${EXTRA_BOOST_LIBS}
        ${OpenCV_LIBS}
        ${VTK_LIBRARIES}
)

# Google Test integration
file(GLOB TEST_SOURCES "test/*.cpp")
if(TEST_SOURCES)
    add_executable(runTests
            ${TEST_SOURCES}
            test/reference_test.cpp
            test/optimize_test.cpp
            test/holorunner_test.cpp
            test/gerch_test.cpp
            test/filters_test.cpp
            test/fft_test.cpp
    )
    target_link_libraries(runTests
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
