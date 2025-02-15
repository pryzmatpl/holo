#ifndef CONSTS_HPP
#define CONSTS_HPP

#include <complex>

// All physical and mathematical constants are grouped into a namespace.
namespace constants {
    const std::complex<float> neg_I(0.0f, -1.0f);
    const std::complex<float> _I(0.0f, 1.0f);
    const float pi = 3.14159265358979323846f;
    const float eigen_intensity = 1.0f;
    const float c = 299792458.0f;

    const float holoeye_delta_x = 6.5e-6f;
    const float holoeye_delta_y = 6.5e-6f;

    const float holoeye_width = 1920.0f;
    const float holoeye_height = 1080.0f;
}

// Declare the global distance variable, defined elsewhere.
extern float glob_distance;

#endif // CONSTS_HPP
