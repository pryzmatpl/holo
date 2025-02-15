#ifndef FFTS_HPP
#define FFTS_HPP

// OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

// Boost
#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions.hpp>

// STL
#include <iostream>
#include <string>
#include <fstream>
#include <chrono>
#include <complex>
#include <functional>

// EIT project headers
#include "filters.hpp"
#include "consts.hpp"

// Typedefs
using Point2f = cv::Point_<float>;

namespace eit {

//------------------------------------------------------------------------------
// Enum: HologramType
// Description: Represents the different types of hologram transforms.
//------------------------------------------------------------------------------
enum HologramType {
    RPN,
    DFT,
    FRESNEL_CONV
};

//------------------------------------------------------------------------------
// Function: typetest
// Description: Debug function to test the type of a cv::Mat.
//------------------------------------------------------------------------------
void typetest(cv::Mat& in);

//------------------------------------------------------------------------------
// Class: eit_hologram
// Description: Contains the methods to create and process holograms for the
//              Holoeye modulator.
//------------------------------------------------------------------------------
class eit_hologram {
public:
    // Utility functions
    static void qshow(cv::Mat& in, std::string name);
    cv::Mat remove_average(cv::Mat input);
    cv::Mat tile_to_fhd(cv::Mat& input);
    cv::Mat tile_to_fhd_amp(cv::Mat& input);

    // Reference wavefronts
    cv::Mat holoeye_ref_wavefront_phase(cv::Mat& input);
    cv::Mat holoeye_ref_wavefront_phase2(cv::Mat& input);
    cv::Mat holoeye_ref_wavefront_squareaperture(cv::Mat& input, float distance = 2.0f);
    cv::Mat holoeye_ref_wavefront(cv::Mat inputimage,
                                  Point2f reference_point_source, // 2D point on the z_distance plane
                                  float amplitude,
                                  float z_distance,
                                  float object_plane_angle = constants::pi / 12);
    void holoeye_ref_wavefront_flat_phase(cv::Mat& inputimage,
                                          float object_plane_angle = constants::pi / 12);
    void holoeye_ref_wavefront_focal(cv::Mat& inputimage);
    cv::Mat holoeye_ref_wavefront_sommerfeld(cv::Mat& inputimage, float distance = 2.0f);
    cv::Mat holoeye_chirp(cv::Mat& padded, float distance = 1.0f);

    // Transforms
    cv::Mat holoeye_dft(cv::Mat& input, cv::Mat& imaginary_output); // Hologram from scratch
    cv::Mat holoeye_angular_spectrum(cv::Mat& input, cv::Mat& imaginary_output); // Hologram from scratch
    cv::Mat holoeye_fractional_ft(cv::Mat& input, cv::Mat& imaginary_output, float angle = constants::pi / 2); // Normal Fourier transform
    cv::Mat holoeye_frft(cv::Mat& input, cv::Mat& imaginary_output, float order = 1.0f);
    cv::Mat holoeye_fresnel(cv::Mat& input, cv::Mat& c_data,
                            float distance_holo_to_image = 1.0f,
                            float angle = 0.0f);
    cv::Mat holoeye_rpn(cv::Mat& input, cv::Mat& outputdata);
    cv::Mat holoeye_rpn_no_twin(cv::Mat input, cv::Mat& outputdata);
    cv::Mat holoeye_movavg(cv::Mat& input);
    cv::Mat holoeye_filter_spatial(cv::Mat& input, cv::Mat& spatial);
    cv::Mat holoeye_avg_del(cv::Mat& input);
    cv::Mat holoeye_filter(cv::Mat& input, std::string fil_type);
    cv::Mat holoeye_filter_c1(cv::Mat& input, std::function<void (cv::Mat&)> filter);
    cv::Mat holoeye_convolution_extended(cv::Mat& input, cv::Mat& cplx, float distance = 2.0f);
    cv::Size set_optimal_holosize(cv::Mat& input, cv::Mat& output);
    cv::Mat holoeye_reference(cv::Mat& input, std::string wf_type);
    cv::Mat holoeye_transform(cv::Mat& padded, cv::Mat& adft_data, std::string holotype);
};

} // namespace eit

#endif // FFTS_HPP
