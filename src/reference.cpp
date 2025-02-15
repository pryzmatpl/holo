#include "../include/filters.hpp"
#include "../include/consts.hpp"
#include "../include/fft.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions/sinc.hpp>
#include <iostream>
#include <fstream>
#include <chrono>
#include <complex>
#include <functional>
#include <cmath>

namespace eit {

// Global variable (consider encapsulating/removing globals later)
float wavelength = 632.8e-9f; // He-Ne laser wavelength

//------------------------------------------------------------------------------
// Method: holoeye_movavg
// Description: Computes a simple 3x3 moving average in the neighborhood and
//              subtracts it from each pixel (absolute difference).
//------------------------------------------------------------------------------
cv::Mat eit_hologram::holoeye_movavg(cv::Mat& input) {
    int rows = input.rows;
    int cols = input.cols;
    cv::Mat output = cv::Mat::zeros(input.size(), input.type());

    // Process only inner pixels to avoid border issues.
    for (int row = 1; row < rows - 1; row++) {
        for (int col = 1; col < cols - 1; col++) {
            float sum = 0.0f;
            // Sum over 3x3 neighborhood.
            for (int roi_row = -1; roi_row <= 1; roi_row++) {
                for (int roi_col = -1; roi_col <= 1; roi_col++) {
                    sum += input.at<float>(row + roi_row, col + roi_col);
                }
            }
            float average = sum / 9.0f;
            float pixel_val = input.at<float>(row, col);
            output.at<float>(row, col) = std::fabs(pixel_val - average);
        }
    }
    return output;
}

//------------------------------------------------------------------------------
// Method: holoeye_ref_wavefront_squareaperture
// Description: Applies a square-aperture reference wavefront to the input.
//------------------------------------------------------------------------------
cv::Mat eit_hologram::holoeye_ref_wavefront_squareaperture(cv::Mat& input, float distance) {
    // Compute aperture area based on modulator dimensions.
    float area = constants::holoeye_delta_x * constants::holoeye_width *
                 constants::holoeye_delta_x * constants::holoeye_height;
    float ref_index = 1.44f;
    float effective_distance = (distance == glob_distance) ? distance : glob_distance;
    float wavenum = 2 * constants::pi / wavelength;
    // omega is not used below, so we omit it: float omega = constants::c / wavelength;

    int rows = input.rows;
    int cols = input.cols;
    int rowmid = rows / 2 + 1;
    int colmid = cols / 2 + 1;

    // Process each pixel; assume input is CV_32FC2 (two channels: real & imag)
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            // For each pixel, get its current complex value.
            cv::Scalar_<float> val = input.at<cv::Scalar_<float>>(row, col);
            std::complex<float> pixel(val[0], val[1]);
            // Compute angles based on pixel position (using delta_x for both dimensions).
            float xangle = std::atan(constants::holoeye_delta_x * row / effective_distance);
            float yangle = std::atan(constants::holoeye_delta_x * col / effective_distance);
            std::complex<float> factor = area * (std::exp(constants::neg_I * wavenum * effective_distance) / effective_distance) *
                                         (std::sin(xangle) / (xangle == 0 ? 1.0f : xangle)) *
                                         (std::sin(yangle) / (yangle == 0 ? 1.0f : yangle));
            pixel *= factor;
            // Write back the result.
            val[0] = pixel.real();
            val[1] = pixel.imag();
            input.at<cv::Scalar_<float>>(row, col) = val;
        }
    }
    std::cout << "holoeye_ref_wavefront_squareaperture( ";
    return input;
}

//------------------------------------------------------------------------------
// Method: holoeye_ref_wavefront_flat_phase
// Description: Multiplies the input by a flat phase ramp (constant phase shift).
//------------------------------------------------------------------------------
void eit_hologram::holoeye_ref_wavefront_flat_phase(cv::Mat& input, float object_plane_angle) {
    cv::Mat planes[2];
    if (input.channels() == 2) {
        cv::split(input, planes);
    } else {
        planes[0] = input;
        planes[1] = cv::Mat::zeros(input.size(), CV_32F);
    }
    int rows = input.rows;
    int cols = input.cols;
    // Apply a constant phase shift to every pixel.
    std::complex<float> phaseFactor = std::polar(1.0f, object_plane_angle);
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            float re = planes[0].at<float>(row, col);
            float im = planes[1].at<float>(row, col);
            std::complex<float> val(re, im);
            val *= phaseFactor;
            planes[0].at<float>(row, col) = val.real();
            planes[1].at<float>(row, col) = val.imag();
        }
    }
    if (input.channels() == 2) {
        cv::merge(planes, 2, input);
    } else {
        input = planes[0];
    }
    std::cout << "holoeye_ref_wavefront_flat_phase( ";
}

//------------------------------------------------------------------------------
// Method: holoeye_ref_wavefront_focal
// Description: Applies a focal (curved) reference phase to the input.
//------------------------------------------------------------------------------
void eit_hologram::holoeye_ref_wavefront_focal(cv::Mat& input) {
    cv::Mat planes[2];
    if (input.channels() == 2) {
        cv::split(input, planes);
    } else {
        planes[0] = input;
        planes[1] = cv::Mat::zeros(input.size(), CV_32F);
    }
    cv::normalize(input, input, 0, 1, cv::NORM_MINMAX);
    int rows = input.rows;
    int cols = input.cols;
    int rowmid = rows / 2 + 1;
    int colmid = cols / 2 + 1;
    // Apply a focal phase factor (example formula; may require tuning)
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            float re = planes[0].at<float>(row, col);
            float im = planes[1].at<float>(row, col);
            std::complex<float> val = std::exp(constants::neg_I * constants::pi *
                                               ((row - rowmid) * (row - rowmid) * constants::holoeye_delta_x) *
                                               ((col - colmid) * (col - colmid) * constants::holoeye_delta_x) /
                                               (wavelength * glob_distance));
            std::complex<float> inp(re, im);
            inp *= val;
            planes[0].at<float>(row, col) = inp.real();
            planes[1].at<float>(row, col) = inp.imag();
        }
    }
    if (input.channels() == 2) {
        cv::merge(planes, 2, input);
    } else {
        input = planes[0];
    }
    std::cout << "holoeye_ref_wavefront_focal( ";
}

//------------------------------------------------------------------------------
// Method: holoeye_ref_wavefront_sommerfeld
// Description: Applies a Rayleigh-Sommerfeld spherical reference wave.
//------------------------------------------------------------------------------
cv::Mat eit_hologram::holoeye_ref_wavefront_sommerfeld(cv::Mat& input, float distance) {
    cv::Mat planes[2];
    if (input.channels() == 2) {
        cv::split(input, planes);
    } else {
        planes[0] = input;
        planes[1] = cv::Mat::zeros(input.size(), CV_32F);
    }
    int rows = planes[0].rows;
    int cols = planes[0].cols;
    int rowmid = rows / 2 + 1;
    int colmid = cols / 2 + 1;
    // Radius of curvature influences the image significantly.
    float RC = glob_distance * std::sqrt(constants::holoeye_width * constants::holoeye_height *
                                           constants::holoeye_delta_x * constants::holoeye_delta_x);
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            float re = planes[0].at<float>(row, col);
            float im = planes[1].at<float>(row, col);
            // Compute a transfer function using a Rayleigh-Sommerfeld formulation.
            std::complex<float> test = (constants::_I * RC) *
                std::exp(constants::_I * 2.0f * constants::pi /
                         (wavelength * std::sqrt(RC * RC +
                          (row - rowmid) * (row - rowmid) * constants::holoeye_delta_x * constants::holoeye_delta_x +
                          (col - colmid) * (col - colmid) * constants::holoeye_delta_x * constants::holoeye_delta_x))) /
                (RC * RC +
                 (row - rowmid) * (row - rowmid) * constants::holoeye_delta_x * constants::holoeye_delta_x +
                 (col - colmid) * (col - colmid) * constants::holoeye_delta_x * constants::holoeye_delta_x);
            // Use the phase of 'test'
            std::complex<float> vas = std::polar(1.0f, std::atan(test.imag() / test.real()));
            std::complex<float> imp(re, im);
            imp *= vas;
            planes[0].at<float>(row, col) = imp.real();
            planes[1].at<float>(row, col) = imp.imag();
        }
    }
    std::cout << "holoeye_ref_wavefront_sommerfeld( ";
    if (input.channels() == 2)
        cv::merge(planes, 2, input);
    else
        input = planes[0];
    return input;
}

//------------------------------------------------------------------------------
// Method: holoeye_ref_wavefront_phase
// Description: Applies a phase modulation based on the column index.
//------------------------------------------------------------------------------
cv::Mat eit_hologram::holoeye_ref_wavefront_phase(cv::Mat& input) {
    cv::Mat planes[2];
    if (input.channels() == 2) {
        cv::split(input, planes);
    } else {
        planes[0] = input;
        planes[1] = cv::Mat::zeros(input.size(), CV_32F);
    }
    int rows = planes[0].rows;
    int cols = planes[0].cols;
    // Apply a phase factor that depends on the column.
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            float re = planes[0].at<float>(row, col);
            float im = planes[1].at<float>(row, col);
            std::complex<float> phaseFactor = std::exp(constants::neg_I * constants::pi / wavelength *
                                                       (col * col * constants::holoeye_delta_x));
            std::complex<float> number(re, im);
            number *= phaseFactor;
            planes[0].at<float>(row, col) = number.real();
            planes[1].at<float>(row, col) = number.imag();
        }
    }
    std::cout << "holoeye_ref_wavefront_phase( ";
    if (input.channels() == 2)
        cv::merge(planes, 2, input);
    else
        input = planes[0];
    return input;
}

//------------------------------------------------------------------------------
// Method: holoeye_ref_wavefront_phase2
// Description: An alternative phase modulation; returns a single-channel phase image.
//------------------------------------------------------------------------------
cv::Mat eit_hologram::holoeye_ref_wavefront_phase2(cv::Mat& input) {
    cv::Mat output = input.clone();
    int rows = input.rows;
    int cols = input.cols;
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            float img_val = input.at<float>(row, col);
            std::complex<float> phaseFactor = std::exp(constants::neg_I * constants::pi / wavelength *
                                                       std::sqrt(col * col * constants::holoeye_delta_x * constants::holoeye_delta_x));
            // Reconstruct a complex number with magnitude = img_val and phase from phaseFactor.
            std::complex<float> number = std::polar(img_val, std::arg(phaseFactor));
            // Store only the phase (argument).
            output.at<float>(row, col) = std::arg(number);
        }
    }
    std::cout << "holoeye_ref_wavefront_phase2( ";
    return output;
}

//------------------------------------------------------------------------------
// Method: holoeye_chirp
// Description: Applies a chirp function (phase modulation) to the input.
//------------------------------------------------------------------------------
cv::Mat eit_hologram::holoeye_chirp(cv::Mat& input, float distance_) {
    cv::Mat planes[2];
    if (input.channels() == 2) {
        cv::split(input, planes);
    } else {
        planes[0] = input;
        planes[1] = cv::Mat::zeros(input.size(), CV_32F);
    }
    cv::normalize(input, input, 0, 1, cv::NORM_MINMAX);
    int rows = input.rows;
    int cols = input.cols;
    int rowmid = rows / 2 + 1;
    int colmid = cols / 2 + 1;
    float effective_distance = (distance_ == glob_distance) ? distance_ : glob_distance;
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            std::complex<float> phaseFactor = std::exp(
                constants::neg_I * 2.0f * constants::pi / (wavelength * effective_distance) *
                ((row - rowmid) * (row - rowmid) * constants::holoeye_delta_x * constants::holoeye_delta_x +
                 (col - colmid) * (col - colmid) * constants::holoeye_delta_x * constants::holoeye_delta_x)
            );
            float re = planes[0].at<float>(row, col);
            float im = planes[1].at<float>(row, col);
            std::complex<float> compVal(re, im);
            compVal *= phaseFactor;
            planes[0].at<float>(row, col) = compVal.real();
            planes[1].at<float>(row, col) = compVal.imag();
        }
    }
    if (input.channels() == 2)
        cv::merge(planes, 2, input);
    else
        input = planes[0];
    std::cout << "holoeye_chirp( ";
    return input;
}

//------------------------------------------------------------------------------
// Method: holoeye_reference
// Description: Selects and applies a reference wavefront type to the input.
//------------------------------------------------------------------------------
cv::Mat eit_hologram::holoeye_reference(cv::Mat& input, std::string wf_type) {
    try {
        cv::normalize(input, input, 0, 1, cv::NORM_MINMAX);
        if (wf_type == "FLAT") {
            holoeye_ref_wavefront_flat_phase(input);
            std::cout << "holoeye_ref_wavefront_flat_phase (";
            qshow(input, "FLAT");
        } else if (wf_type == "REF") {
            holoeye_ref_wavefront_phase(input);
            std::cout << "holoeye_ref_wavefront_phase (";
            qshow(input, "REF");
        } else if (wf_type == "REFFOCAL") {
            holoeye_ref_wavefront_focal(input);
            std::cout << "holoeye_ref_wavefront_focal (";
            qshow(input, "REFFOCAL");
        } else if (wf_type == "SQUARE") {
            holoeye_ref_wavefront_squareaperture(input);
            std::cout << "holoeye_ref_wavefront_squareaperture (";
            qshow(input, "SQUARE");
        } else if (wf_type == "REF2") {
            holoeye_ref_wavefront_phase2(input);
            std::cout << "holoeye_ref_wavefront_phase2 (";
            qshow(input, "REF2");
        } else if (wf_type == "RAYLEIGH") {
            holoeye_ref_wavefront_sommerfeld(input);
            std::cout << "holoeye_ref_wavefront_sommerfeld (";
            qshow(input, "RAYLEIGH");
        } else if (wf_type == "CHIRP") {
            holoeye_chirp(input);
            std::cout << "holoeye_chirp (";
            qshow(input, "CHIRP");
        }
        cv::normalize(input, input, 0, 1, cv::NORM_MINMAX);
        std::cout << std::flush;
        return input;
    } catch (std::exception &e) {
        std::cout << "Exception in holoeye_reference: " << e.what() << "\n";
        return input;
    }
}

} // namespace eit
