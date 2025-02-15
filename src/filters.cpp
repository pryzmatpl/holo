#include "../include/filters.hpp"
#include "../include/fft.hpp"
#include <boost/math/special_functions/sinc.hpp> // For boost::math::sinc_pi
#include <cmath>
#include <cstdlib>

// External global variable (defined elsewhere)
extern float wavelength; // TODO: Consider encapsulating/removing globals.

namespace eit {

//------------------------------------------------------------------------------
// Member function: holoeye_filter_c1
// Applies the given filter (a std::function) to a copy of the input.
//------------------------------------------------------------------------------
cv::Mat eit_hologram::holoeye_filter_c1(cv::Mat& input, std::function<void(cv::Mat&)> filter) {
    cv::Mat returnee(input);
    filter(returnee);
    return returnee;
}

//------------------------------------------------------------------------------
// Filter lambdas
// (These definitions correspond to the extern declarations in filters.hpp)
//------------------------------------------------------------------------------

// Checker filter: Zeroes or adjusts pixels in a centered square region.
holo_filter filter_checker = [](cv::Mat& in) {
    cv::Mat returnee[2] = { cv::Mat_<float>(in.size(), in.type()),
                            cv::Mat::zeros(in.size(), in.type()) };

    if (in.channels() == 2)
        cv::split(in, returnee);
    else
        returnee[0] = in;

    int rowmid = in.rows / 2 + 1;
    int colmid = in.cols / 2 + 1;

    for (int row = 0; row < in.rows; row++) {
        for (int col = 0; col < in.cols; col++) {
            if ((std::abs(row - rowmid) + std::abs(col - colmid)) < 70) {
                if ((row + col) % 2 == 0) {
                    if (in.channels() == 2) {
                        float &re = returnee[0].at<float>(row, col);
                        float &im = returnee[1].at<float>(row, col);
                        re = 0.0f;
                        im = 0.0f;
                    } else {
                        float &val = returnee[0].at<float>(row, col);
                        val = 0.0f;
                    }
                } else {
                    if (in.channels() == 2) {
                        float &re = returnee[0].at<float>(row, col);
                        float &im = returnee[1].at<float>(row, col);
                        re += 1.0f;
                        im -= 1.0f;
                    } else {
                        float &val = returnee[0].at<float>(row, col);
                        val += 2.0f;
                    }
                }
            }
        }
    }
    if (in.channels() == 2)
        cv::merge(returnee, 2, in);
    else
        in = returnee[0];
};

// Inverse Fourier kernel filter: Suppresses twin images.
holo_filter filter_inverse_fourier_kernel_one = [](cv::Mat& in) {
    cv::Mat returnee[2] = { cv::Mat_<float>(in.size(), in.type()),
                            cv::Mat::zeros(in.size(), in.type()) };

    if (in.channels() == 2)
        cv::split(in, returnee);
    else
        returnee[0] = in;

    int rowmid = in.rows / 2 + 1;
    int colmid = in.cols / 2 + 1;

    for (int row = 0; row < in.rows; row++) {
        for (int col = 0; col < in.cols; col++) {
            float &pxl_re = returnee[0].at<float>(row, col);
            float &pxl_im = returnee[1].at<float>(row, col);
            std::complex<float> val_filt = 1.0f / (1.0f +
                std::exp(constants::neg_I * 2.0f * constants::pi * wavelength * glob_distance *
                    static_cast<float>((row - rowmid) * (row - rowmid) +
                                       (col - colmid) * (col - colmid))));
            std::complex<float> val_img(pxl_re, pxl_im);
            val_img *= val_filt;
            pxl_re = val_img.real();
            pxl_im = val_img.imag();
        }
    }
    if (in.channels() == 2)
        cv::merge(returnee, 2, in);
    else
        in = returnee[0];
};

// RPN filter: Adds a random phase component.
holo_filter filter_rpn = [](cv::Mat& in) {
    cv::Mat returnee[2] = { cv::Mat_<float>(in.size(), in.type()),
                            cv::Mat::zeros(in.size(), in.type()) };
    cv::split(in, returnee);

    boost::mt19937 rng;
    boost::normal_distribution<> nd(0.0, 1.0);
    boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > var_nor(rng, nd);

    for (int row = 0; row < in.rows; row++) {
        for (int col = 0; col < in.cols; col++) {
            float &pxl_re = returnee[0].at<float>(row, col);
            float &pxl_im = returnee[1].at<float>(row, col);
            float rnd_phase_rad = var_nor(); // Random phase
            std::complex<float> val = std::polar<float>(in.at<float>(row, col), 0);
            std::complex<float> phasevalue(std::exp(constants::neg_I * rnd_phase_rad));
            val += phasevalue;
            pxl_re = val.real();
            pxl_im = val.imag();
        }
    }
    cv::merge(returnee, 2, in);
};

// Spherical filter: Applies a phase shift based on a spherical reference.
holo_filter filter_spherical = [](cv::Mat& in) {
    cv::Mat returnee[2] = { cv::Mat_<float>(in.size(), in.type()),
                            cv::Mat::zeros(in.size(), in.type()) };
    if (in.channels() == 2)
        cv::split(in, returnee);
    else
        returnee[0] = in;

    int rowmid = in.rows / 2 + 1;
    int colmid = in.cols / 2 + 1;

    for (int row = 0; row < in.rows; row++) {
        for (int col = 0; col < in.cols; col++) {
            float &re = returnee[0].at<float>(row, col);
            float &im = returnee[1].at<float>(row, col);
            std::complex<float> pinfo(re, im);
            std::complex<float> paddition = std::polar(1.0f,
                ((2 * constants::pi / wavelength) / glob_distance) *
                ((row - rowmid) * (row - rowmid) + (col - colmid) * (col - colmid)));
            std::complex<float> ret = std::polar(std::sqrt(std::norm(pinfo)),
                std::arg(paddition) + std::arg(pinfo));
            re = ret.real();
            im = ret.imag();
        }
    }
};

// RPN OCV filter: Replaces the image with a random normal distribution.
holo_filter filter_rpn_ocv = [](cv::Mat& in) {
    cv::randn(in, 128, 30); // Magic numbers—consider parameterizing.
};

// EIT LPL filter: Implements an integral image formation for linear phase loading.
holo_filter filter_eit_lpl = [](cv::Mat& in) {
    cv::Mat ret(in.cols + 1, in.rows + 1, in.depth());
    cv::Mat ret_sq(in.rows + 1, in.cols + 1, in.depth());
    cv::Mat ret_tilt(in.rows + 1, in.cols + 1, in.depth());
    cv::integral(in, ret, ret_sq, ret_tilt, in.depth());
    in = cv::Mat(ret_sq, cv::Rect(1, 1, in.cols, in.rows));
};

// Linear phase loading filter.
holo_filter filter_linear_phase_load = [](cv::Mat& in) {
    cv::Mat returnee[2] = { cv::Mat_<float>(in.size(), in.type()),
                            cv::Mat::zeros(in.size(), in.type()) };
    if (in.channels() == 2)
        cv::split(in, returnee);
    else
        returnee[0] = in;

    int rowmid = in.rows / 2 + 1;
    int colmid = in.cols / 2 + 1;
    float factor = 1.0f / 256.0f; // Magic constant
    for (int row = 0; row < in.rows; row++) {
        for (int col = 0; col < in.cols; col++) {
            float &re = returnee[0].at<float>(row, col);
            float &im = returnee[1].at<float>(row, col);
            std::complex<float> pinfo(re, im);
            float aFactor = static_cast<float>(colmid) / static_cast<float>(rowmid);
            float loadedphase = aFactor * (static_cast<float>(row) - rowmid + static_cast<float>(col) - colmid);
            std::complex<float> ret = std::polar(std::sqrt(std::norm(pinfo)), loadedphase + factor * std::arg(pinfo));
            re = ret.real();
            im = ret.imag();
        }
    }
    cv::merge(returnee, 2, in);
};

// Real Laplacian filter: Applies Gaussian blur and then Laplacian.
holo_filter filter_real_laplacian = [](cv::Mat& in) {
    cv::GaussianBlur(in, in, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
    cv::Laplacian(in, in, in.type(), 3, 1, 0, cv::BORDER_DEFAULT);
};

// Clear center filter: Clears the central region of the image.
holo_filter filter_clear_center = [](cv::Mat& in) {
    cv::Mat returnee[2] = { cv::Mat_<float>(in.size(), in.type()),
                            cv::Mat::zeros(in.size(), in.type()) };
    cv::split(in, returnee);
    int rowmid = returnee[0].rows / 2 + 1;
    int colmid = returnee[0].cols / 2 + 1;
    int border = static_cast<int>(0.1 * returnee[0].rows);
    for (int row = rowmid - border; row < rowmid + border; row++) {
        for (int col = colmid - border; col < colmid + border; col++) {
            float &re = returnee[0].at<float>(row, col);
            float &im = returnee[1].at<float>(row, col);
            if (std::abs((row - rowmid) + (col - colmid)) < border) {
                re = 1.0f;
                im = std::sqrt(((row - rowmid) * (row - rowmid)) + ((col - colmid) * (col - colmid)) * wavelength * wavelength);
            }
        }
    }
    cv::merge(returnee, 2, in);
};

// Sinc filter: Multiplies each pixel by a sinc function computed from its distance to the center.
holo_filter filter_sinc = [](cv::Mat& in) {
    int rowmid = in.rows / 2 + 1;
    int colmid = in.cols / 2 + 1;
    for (eit::row aRow = 0; aRow < in.rows; aRow++) {
        for (eit::col aCol = 0; aCol < in.cols; aCol++) {
            float flx = boost::math::sinc_pi<float>(
                ((2 * constants::pi) / (wavelength * glob_distance)) *
                ((static_cast<float>((aRow - rowmid) * (aRow - rowmid)) * constants::holoeye_delta_x * constants::holoeye_delta_x) +
                 (static_cast<float>((aCol - colmid) * (aCol - colmid)) * constants::holoeye_delta_x * constants::holoeye_delta_x))
            );
            std::complex<float> sinc_arg = std::polar(1.0f, flx);
            cv::Vec3f& pxl = in.at<cv::Vec3f>(aRow, aCol);
            std::complex<float> value(pxl[0], (in.channels() == 1 ? 0.0f : pxl[1]));
            value *= sinc_arg;
            pxl[0] = value.real();
            pxl[1] = (in.channels() == 1 ? 0.0f : value.imag());
        }
    }
};

// REMAVG filter: Removes the average (mean) value from the image.
// Assumes single-channel CV_32F.
holo_filter filter_remavg = [](cv::Mat& in) {
    cv::Scalar mean_ = cv::mean(in);
    for (int row = 0; row < in.rows; row++) {
        for (int col = 0; col < in.cols; col++) {
            in.at<float>(row, col) -= static_cast<float>(mean_[0]);
        }
    }
};

//------------------------------------------------------------------------------
// Member function: holoeye_filter
// Selects and applies one of the filters based on the string fil_type.
//------------------------------------------------------------------------------
cv::Mat eit_hologram::holoeye_filter(cv::Mat& input, std::string fil_type) {
    try {
        if (fil_type == "REMAVG") {
            std::cout << "filter_remavg <";
            holoeye_filter_c1(input, filter_remavg);
        } else if (fil_type == "SINC") {
            std::cout << "filter_sinc <";
            holoeye_filter_c1(input, filter_sinc);
        } else if (fil_type == "LAPLACIAN") {
            std::cout << "filter_real_laplacian <";
            holoeye_filter_c1(input, filter_real_laplacian);
        } else if (fil_type == "SPHERICAL") {
            std::cout << "filter_spherical <";
            holoeye_filter_c1(input, filter_spherical);
        } else if (fil_type == "RPN") {
            std::cout << "filter_rpn <";
            holoeye_filter_c1(input, filter_rpn);
        } else if (fil_type == "LINEAR") {
            std::cout << "filter_linear <";
            holoeye_filter_c1(input, filter_linear_phase_load);
        } else if (fil_type == "EIT_LPL") {
            std::cout << "filter_eit_lpl <";
            holoeye_filter_c1(input, filter_eit_lpl);
        } else if (fil_type == "RPNOCV") {
            std::cout << "filter_rpn_ocv <";
            holoeye_filter_c1(input, filter_rpn_ocv);
        } else if (fil_type == "TWINREM") {
            std::cout << "filter_inverse_fourier_kernel_one <";
            holoeye_filter_c1(input, filter_inverse_fourier_kernel_one);
        } else if (fil_type == "CLEARCENTER") {
            std::cout << "filter_clear_center <";
            holoeye_filter_c1(input, filter_clear_center);
        } else if (fil_type == "NONE") {
            std::cout << "(filimage=NONE)";
        } else if (fil_type == "CHECKER") {
            std::cout << "(filimage=CHECKER)";
            holoeye_filter_c1(input, filter_checker);
        }
        std::cout << std::flush;
        return input;
    } catch (std::exception &e) {
        std::cout << "Exception in holoeye_filter: " << e.what() << "\n";
    }
    return input;
}

//------------------------------------------------------------------------------
// Member function: holoeye_filter_spatial
// Applies a spatial filter in the frequency domain.
//------------------------------------------------------------------------------
cv::Mat eit_hologram::holoeye_filter_spatial(cv::Mat& input, cv::Mat& spatial_filter) {
    cv::Mat returnee[2] = { cv::Mat_<float>(input.size(), input.type()),
                            cv::Mat::zeros(input.size(), input.type()) };
    input.copyTo(returnee[0]);

    cv::Mat filter_parts[2] = { cv::Mat_<float>(input.size(), input.type()),
                                cv::Mat::zeros(input.size(), input.type()) };

    cv::resize(spatial_filter, filter_parts[0], input.size());
    filter_parts[0].convertTo(filter_parts[0], CV_32FC2);

    cv::Mat ret_dft(input.size(), CV_32FC2);
    cv::Mat filter_dft(input.size(), CV_32FC2);

    cv::merge(returnee, 2, ret_dft);
    cv::dft(ret_dft, ret_dft);

    cv::merge(filter_parts, 2, filter_dft);
    cv::dft(filter_dft, filter_dft);

    cv::split(ret_dft, returnee);
    cv::split(filter_dft, filter_parts);

    try {
        for (int row = 0; row < input.rows; row++) {
            for (int col = 0; col < input.cols; col++) {
                std::complex<float> final_pxl_value(returnee[0].at<float>(row, col),
                                                    returnee[1].at<float>(row, col));
                std::complex<float> filter_pxl_value(filter_parts[0].at<float>(row, col),
                                                     filter_parts[1].at<float>(row, col));
                final_pxl_value *= filter_pxl_value;
                returnee[0].at<float>(row, col) = final_pxl_value.real();
                returnee[1].at<float>(row, col) = final_pxl_value.imag();
            }
        }
        cv::merge(returnee, 2, ret_dft);
        cv::dft(ret_dft, ret_dft, cv::DFT_INVERSE);
        cv::split(ret_dft, returnee);
        cv::phase(returnee[0], returnee[1], returnee[0]);
    } catch (cv::Exception& e) {
        std::cout << "Official error:" << e.what() << std::endl;
        std::exit(-1);
    }
    return returnee[0];
}

} // namespace eit
