# Prizm Holo - Holographic Computation with OpenCV and Boost

## Overview

Holo is a C++ project designed to generate holograms using OpenCV and Boost libraries. It provides multiple algorithms for hologram generation, including Fresnel, RPN, FFT, and fractional Fourier transforms, with additional filtering capabilities for image processing.

This project was developed for use with the **Holoeye LCoS** spatial light modulator and supports various reference wavefront transformations, including:

- Flat phase reference
- Fresnel transformation
- Sommerfeld-Rayleigh approximation
- Gerchberg-Saxton algorithm (experimental)

## Features

- **Multiple holographic transformation methods** (Fresnel, RPN, FFT, etc.)
- **Filtering capabilities** (Sinc, Laplacian, twin removal, and more)
- **Reference wavefront customization** (Flat, Chirp, Rayleigh, etc.)
- **Real-time OpenCV visualization**
- **Boost-based CLI options for flexibility**
- **Thread-safe operations using mutex locks**

## Dependencies

To build and run Holo, ensure you have the following dependencies installed:

- **C++11 or later**
- **CMake (>=2.6)**
- **Boost (>=1.55.0)**
  - Components: `system`, `thread`, `iostreams`, `program_options`
- **OpenCV (>=2.4.10)**
- **GNU Scientific Library (GSL)**

## Directory Structure

```
Holo/
├── assets/         # Additional assets for testing
├── cmake/          # CMake configuration files
├── include/        # Header files for FFT, filtering, etc.
├── src/            # Source files for hologram processing
├── .git/           # Git repository metadata
├── CMakeLists.txt  # CMake build configuration
├── LICENSE         # License file
├── main.cpp        # Main entry point
├── main.hpp        # Core header file
├── README.md       # This file
└── TODO            # Planned improvements
```

## Compilation and Installation

To compile Holo, follow these steps:

```sh
mkdir build && cd build
cmake ..
make -j$(nproc)
```

This will generate the following binaries:

- **holo** - Main hologram generation executable
- **holorunner** - Test runner for alternating phase images

## Usage

Run the program with:

```sh
./holo --image <input_image> --distance <float> --type <FRESNEL/RPN/FFT>
```

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

## Future Improvements

- Pipeline updates
- Possibly adding more LCoS

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

Special thanks to the OpenCV and Boost communities for their powerful tools in image processing and CLI argument parsing.

