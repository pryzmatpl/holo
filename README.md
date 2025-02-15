# Prizm Holo – Advanced Holographic Computation

## Overview

Prizm Holo is a C++ project for generating holograms using advanced algorithms and leveraging popular libraries such as OpenCV, Boost, and VTK. Originally developed for the Holoeye LCoS spatial light modulator, the project now uses C++17 features and has been refactored to improve configurability, modularity, and testability.

The application supports several holographic transformation methods including:
- **Fresnel Transformation**
- **Random Phase (RPN) Method**
- **Fast Fourier Transform (FFT)**
- **Fractional Fourier Transform (FRFT)**
- **Angular Spectrum Method**

## Features

- **Multiple Holographic Algorithms:** Implementations of Fresnel, RPN, FFT, FRFT, and angular spectrum transforms.
- **Configurable Reference Wavefronts:** Options include flat phase, chirp, Rayleigh, and square aperture references.
- **Modern C++ Design:** Uses C++17 with clean code separation and a configuration structure (e.g., `HologramOptions`) to eliminate global variables.
- **Modular Build System:** Core processing code is packaged as a library (`holo_core`) used by both main executables and tests.
- **Robust Testing Framework:** Comprehensive unit tests are implemented using Google Test.
- **VTK & GDAL Integration:** Optional integration with VTK for visualization and GDAL compatibility handling on Linux.
- **Custom Run Script:** A shell script (`run.sh`) is provided to set runtime environment variables (e.g. `LD_LIBRARY_PATH`, `LD_PRELOAD`) and select which executable to run with additional command-line arguments.


## Dependencies

To build and run Prizm Holo, ensure that you have the following dependencies installed:

- **C++17 Compiler** (GCC 7.0+ recommended)
- **CMake (>=3.10)**
- **Boost (>=1.55.0)**
  - Required components: `system`, `thread`, `iostreams`, `program_options`
- **OpenCV (>=4.11.0)**
- **GNU Scientific Library (GSL)**
- **VTK** (for visualization and GDAL support; ensure GDAL is installed or disable VTK GDAL support)
- **CLI11** (automatically fetched via CMake if not available)
- **Google Test** (for running the test suite)

## Directory Structure

```
PrizmHolo/ 
├── assets/ # Sample images and test assets 
├── cmake/ # Custom CMake modules 
├── include/ # Project headers (FFT, filtering, options, etc.) 
├── src/ # Source files for hologram processing and configuration 
│ ├── reference.cpp 
│ ├── fft.cpp 
│ ├── filters.cpp 
│ ├── gerch.cpp 
│ └── optimize.cpp # Helper functions 
├── test/ # Unit tests and test main (Google Test) 
│ ├── main.cpp # Single main() for tests 
│ ├── fft_test.cpp 
│ ├── filters_test.cpp 
│ ├── gerch_test.cpp 
│ ├── optimize_test.cpp 
│ └── reference_test.cpp 
├── CMakeLists.txt # Build configuration (creates holo_core library) 
├── run.sh # Shell script to run executables with proper environment variables 
├── README.md # This file 
└── LICENSE # License file (MIT)
```

## Compilation and Installation

To compile Holo, follow these steps:

```sh
mkdir build && cd build
cmake ..
make -j$(nproc)
```

This configuration will:
* Set the C++ standard to C++17.
* Automatically fetch CLI11 if needed.
* Find and configure Boost, OpenCV, GSL, VTK, and Google Test.
* Build the core library (holo_core) and main executables (holo, holorunner).

## Usage
Run Script

A helper shell script (run.sh) is provided to set the required environment variables (such as LD_LIBRARY_PATH and LD_PRELOAD) and to select which executable to run, with the ability to pass additional command-line arguments.

Example usage:

./run.sh holo --image sample.jpg --distance 1.5 --type FRESNEL --refwave REF
./run.sh holorunner --maskfile pixel_list.txt --frequency 10
./run.sh tests --gtest_filter=MyTest*

## Running Tests

To run the full test suite:

> ./runTests

Or, using CTest:

> ctest


### Command-Line Options

| Option         | Description                                              |
| -------------- | -------------------------------------------------------- |
| `--help`       | Show available options                                   |
| `--image`      | Path to input image (default: `_logo.jpg`)               |
| `--distance`   | Distance from SLM to hologram                            |
| `--type`       | Hologram type: `FRESNEL`, `RPN`, `FFT`, etc.             |
| `--gerch`      | Use Gerchberg-Saxton algorithm (true/false)              |
| `--refwave`    | Reference wavefront type (FLAT, REF, CHIRP, etc.)        |
| `--filimage`   | Apply filter to real input image (SINC, LAPLACIAN, etc.) |
| `--wavelength` | Set wavelength (default: 632.8e-9)                       |
| `--frft_angle` | Fractional Fourier transform angle                       |
| `--chirp`      | Apply chirp function post-FFT                            |

## Example

To generate a hologram using a Fresnel transformation:

```sh
./holo --image sample.jpg --distance 1.5 --type FRESNEL --refwave REF
```

To alternate images using `holorunner`:

```sh
./holorunner --maskfile pixel_list.txt --frequency 10
```

## GDAL and VTK

On Arch Linux (and similar distributions), ensure that GDAL is installed and that VTK is built against a compatible GDAL version. If you encounter linking errors related to GDAL, you may need to:

    Install the correct GDAL package (e.g., sudo pacman -S gdal).
    Adjust your LD_LIBRARY_PATH or use RPATH settings to locate the correct GDAL library.
    Alternatively, disable GDAL support in VTK if your application doesn’t require it.

## Future Improvements

    Further decoupling of configuration using a dedicated options class.
    Expanding the range of holographic algorithms and filters.
    Enhancing real-time visualization.
    Improved error handling and logging.
    Potential integration with deep learning for adaptive hologram generation.

## License

This project is licensed under the PRIZM - MIT License. See the `LICENSE` file for details.

## Acknowledgments

Special thanks to the OpenCV and Boost communities for their powerful tools in image processing and CLI argument parsing.

