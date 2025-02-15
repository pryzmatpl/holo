#ifndef FILTERS_HPP
#define FILTERS_HPP

// OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

// Boost
#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>
#include <boost/tokenizer.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions.hpp>
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>

// STL
#include <iostream>
#include <string>
#include <fstream>
#include <chrono>
#include <complex>
#include <functional>

// Project headers
#include "consts.hpp"

namespace eit {

    // Typedef for a hologram filter function that operates on a cv::Mat.
    using holo_filter = std::function<void(cv::Mat&)>;

    // Typedef for a cv::Mat iterator over cv::Vec3f elements.
    using Iterator = cv::MatIterator_<cv::Vec3f>;

    // Typedefs for row and column indices.
    using row = unsigned int;
    using col = unsigned int;

    // External filter function declarations.
    // These filters should be defined in a corresponding .cpp file.
    extern holo_filter filter_remavg;
    extern holo_filter filter_sinc;
    extern holo_filter filter_rpn;
    extern holo_filter filter_rpn_ocv;
    extern holo_filter filter_laplacian;
    extern holo_filter filter_clear_center;
    extern holo_filter filter_real_laplacian;
    extern holo_filter filter_inverse_fourier_kernel_one;
    extern holo_filter filter_spherical;
    extern holo_filter filter_checker;
    extern holo_filter filter_eit_lpl;
    extern holo_filter filter_linear_phase_load;

} // namespace eit

#endif // FILTERS_HPP
