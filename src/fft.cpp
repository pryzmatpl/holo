#include "../include/fft.hpp"

// Based on "Fast algorithms for free-space diffraction patterns calculation"
// by David Mas et. al.
// We CANNOT use Fraunhoffer diffraction, due to the fact that the conditions
// are not met! Our image and source would have to be 81 m apart from the aperture.
// But why not test it? We treat our modulator as an aperture, with the laser acting
// as a source that will be well aligned, just so that we will be able to calculate the hologram.

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <complex>
#include <iostream>
#include <valarray>
#include <algorithm>
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/thread.hpp>

// External globals (defined elsewhere in your project)
extern float wavelength; // TODO: rework global if possible
extern float glob_distance;
extern float frft_angle; // Global fractional FT angle

// -----------------------------------------------------------------------------
// Global operator overloads for cv::Mat and std::complex<float>
// -----------------------------------------------------------------------------

cv::Mat& operator*(cv::Mat& lhs, std::complex<float> rhs) {
    for (int row = 0; row < lhs.rows; row++) {
        for (int col = 0; col < lhs.cols; col++) {
            cv::Vec3b& x = lhs.at<cv::Vec3b>(row, col);
            if (lhs.channels() == 1) {
                x[0] *= rhs.real();
            }
            else if (lhs.channels() == 2) {
                x[0] *= rhs.real();
                x[1] *= rhs.imag();
            }
            else {
                throw std::runtime_error("Channels do not match in operator*(Mat&, complex<float>)");
            }
        }
    }
    return lhs;
}

cv::Mat& operator*=(cv::Mat& lhs, std::complex<float> rhs) {
    for (int row = 0; row < lhs.rows; row++) {
        for (int col = 0; col < lhs.cols; col++) {
            cv::Vec3b& x = lhs.at<cv::Vec3b>(row, col);
            if (lhs.channels() == 1) {
                x[0] *= rhs.real();
            }
            else if (lhs.channels() == 2) {
                x[0] *= rhs.real();
                x[1] *= rhs.imag();
            }
            else {
                throw std::runtime_error("Channels do not match in operator*=(Mat&, complex<float>)");
            }
        }
    }
    return lhs;
}

bool operator<(const cv::Size& lhs, const cv::Size& rhs) {
    return (lhs.width * lhs.height) < (rhs.width * rhs.height);
}

cv::Mat operator*(cv::Mat& lhs, cv::Mat& rhs) {
    // TODO: Ensure compatibility when matrices differ in channels or size.
    cv::Mat result = cv::Mat::zeros(lhs.size(), lhs.type());
    for (int row = 0; row < lhs.rows; row++) {
        for (int col = 0; col < lhs.cols; col++) {
            cv::Vec3b& rVal = result.at<cv::Vec3b>(row, col);
            cv::Vec3b& lhsVal = lhs.at<cv::Vec3b>(row, col);
            cv::Vec3b& rhsVal = rhs.at<cv::Vec3b>(row, col);
            for (int ch = 0; ch < rVal.channels; ch++) {
                rVal[ch] = lhsVal[ch] * rhsVal[ch];
            }
        }
    }
    return result;
}

// -----------------------------------------------------------------------------
// Begin definitions in the eit namespace (methods of eit_hologram, etc.)
// -----------------------------------------------------------------------------
namespace eit {

void typetest(cv::Mat& in) {
    std::cout << in.size() << ", " << in.type() << ", " << in.channels() << std::endl;
}

// -----------------------------------------------------------------------------
// Tiling functions
// -----------------------------------------------------------------------------

cv::Mat eit_hologram::tile_to_fhd(cv::Mat& input) {
    try {
        cv::Mat magI;
        if (input.channels() != 1) {
            cv::Mat planes[] = { cv::Mat::zeros(input.size(), CV_32F),
                                 cv::Mat::zeros(input.size(), CV_32F) };
            cv::split(input, planes);
            cv::phase(planes[0], planes[1], magI);
        } else {
            magI = input;
        }

        magI += cv::Scalar::all(1); // switch to logarithmic scale
        // Crop the spectrum if it has an odd number of rows or columns.
        magI = magI(cv::Rect(0, 0, magI.cols & -2, magI.rows & -2));

        // Rearrange the quadrants so that the origin is at the center.
        int cx = magI.cols / 2;
        int cy = magI.rows / 2;
        cv::Mat q0(magI, cv::Rect(0, 0, cx, cy));   // Top-Left
        cv::Mat q1(magI, cv::Rect(cx, 0, cx, cy));    // Top-Right
        cv::Mat q2(magI, cv::Rect(0, cy, cx, cy));    // Bottom-Left
        cv::Mat q3(magI, cv::Rect(cx, cy, cx, cy));    // Bottom-Right
        cv::Mat tmp;
        q0.copyTo(tmp);
        q3.copyTo(q0);
        tmp.copyTo(q3);
        q1.copyTo(tmp);
        q2.copyTo(q1);
        tmp.copyTo(q2);

        cv::normalize(magI, magI, 0, 1, cv::NORM_MINMAX);

        cv::Mat tiled(cv::Size(static_cast<int>(constants::holoeye_width),
                               static_cast<int>(constants::holoeye_height)),
                      magI.type());

        // Tile the image across the new full-HD resolution.
        for (int row = 0; row < tiled.rows; row++) {
            for (int col = 0; col < tiled.cols; col++) {
                if ((row % input.rows == 0) && (col % input.cols == 0)) {
                    if ((row + input.rows > constants::holoeye_height) &&
                        (col + input.cols > constants::holoeye_width)) {
                        cv::Mat roi(tiled, cv::Rect(col, row,
                            static_cast<int>(constants::holoeye_width) - col,
                            static_cast<int>(constants::holoeye_height) - row));
                        cv::Mat roi_magI(magI, cv::Rect(0, 0,
                            static_cast<int>(constants::holoeye_width) - col,
                            static_cast<int>(constants::holoeye_height) - row));
                        roi_magI.copyTo(roi);
                    }
                    else if ((col + magI.cols > constants::holoeye_width)) {
                        cv::Mat roi(tiled, cv::Rect(col, row,
                            static_cast<int>(constants::holoeye_width) - col,
                            magI.rows));
                        cv::Mat roi_magI(magI, cv::Rect(0, 0,
                            static_cast<int>(constants::holoeye_width) - col,
                            magI.rows));
                        roi_magI.copyTo(roi);
                    }
                    else if ((row + magI.rows) > constants::holoeye_height) {
                        cv::Mat roi(tiled, cv::Rect(col, row,
                            magI.cols,
                            static_cast<int>(constants::holoeye_height) - row));
                        cv::Mat roi_magI(magI, cv::Rect(0, 0,
                            magI.cols,
                            static_cast<int>(constants::holoeye_height) - row));
                        roi_magI.copyTo(roi);
                    }
                    else {
                        cv::Mat roi(tiled, cv::Rect(col, row, magI.cols, magI.rows));
                        magI.copyTo(roi);
                    }
                }
            }
        }
        cv::normalize(tiled, tiled, 0, 1, cv::NORM_MINMAX);
        return tiled;
    }
    catch (std::exception& e) {
        std::cout << "Error in tile_to_fhd - " << e.what() << "\n";
        return cv::Mat();
    }
}

cv::Mat eit_hologram::tile_to_fhd_amp(cv::Mat& input) {
    try {
        cv::Mat planes[2] = { cv::Mat::zeros(input.size(), CV_32F),
                              cv::Mat::zeros(input.size(), CV_32F) };
        cv::split(input, planes);

        cv::Mat magI;
        cv::magnitude(planes[0], planes[1], magI);

        magI += cv::Scalar::all(1);
        magI = magI(cv::Rect(0, 0, magI.cols & -2, magI.rows & -2));

        int cx = magI.cols / 2;
        int cy = magI.rows / 2;

        cv::Mat q0(magI, cv::Rect(0, 0, cx, cy));
        cv::Mat q1(magI, cv::Rect(cx, 0, cx, cy));
        cv::Mat q2(magI, cv::Rect(0, cy, cx, cy));
        cv::Mat q3(magI, cv::Rect(cx, cy, cx, cy));
        cv::Mat tmp;
        q0.copyTo(tmp);
        q3.copyTo(q0);
        tmp.copyTo(q3);
        q1.copyTo(tmp);
        q2.copyTo(q1);
        tmp.copyTo(q2);

        cv::normalize(magI, magI, 0, 1, cv::NORM_MINMAX);

        cv::Mat tiled(cv::Size(static_cast<int>(constants::holoeye_width),
                               static_cast<int>(constants::holoeye_height)),
                      magI.type());

        for (int row = 0; row < tiled.rows; row++) {
            for (int col = 0; col < tiled.cols; col++) {
                if ((row % input.rows == 0) && (col % input.cols == 0)) {
                    if ((row + input.rows > constants::holoeye_height) &&
                        (col + input.cols > constants::holoeye_width)) {
                        cv::Mat roi(tiled, cv::Rect(col, row,
                            static_cast<int>(constants::holoeye_width) - col,
                            static_cast<int>(constants::holoeye_height) - row));
                        cv::Mat roi_magI(magI, cv::Rect(0, 0,
                            static_cast<int>(constants::holoeye_width) - col,
                            static_cast<int>(constants::holoeye_height) - row));
                        roi_magI.copyTo(roi);
                    }
                    else if ((col + magI.cols > constants::holoeye_width)) {
                        cv::Mat roi(tiled, cv::Rect(col, row,
                            static_cast<int>(constants::holoeye_width) - col,
                            magI.rows));
                        cv::Mat roi_magI(magI, cv::Rect(0, 0,
                            static_cast<int>(constants::holoeye_width) - col,
                            magI.rows));
                        roi_magI.copyTo(roi);
                    }
                    else if ((row + magI.rows) > constants::holoeye_height) {
                        cv::Mat roi(tiled, cv::Rect(col, row,
                            magI.cols,
                            static_cast<int>(constants::holoeye_height) - row));
                        cv::Mat roi_magI(magI, cv::Rect(0, 0,
                            magI.cols,
                            static_cast<int>(constants::holoeye_height) - row));
                        roi_magI.copyTo(roi);
                    }
                    else {
                        cv::Mat roi(tiled, cv::Rect(col, row, magI.cols, magI.rows));
                        magI.copyTo(roi);
                    }
                }
            }
        }
        cv::normalize(tiled, tiled, 0, 1, cv::NORM_MINMAX);
        return tiled;
    }
    catch (std::exception& e) {
        std::cout << "Error in tile_to_fhd_amp - " << e.what() << "\n";
        return cv::Mat();
    }
}

// -----------------------------------------------------------------------------
// set_optimal_holosize: Prepares a real image by normalizing, padding, and
// converting it to a two-channel (complex) image.
// -----------------------------------------------------------------------------
cv::Size eit_hologram::set_optimal_holosize(cv::Mat& input, cv::Mat& output) {
    try {
        cv::normalize(input, input, 0, 1, cv::NORM_MINMAX);
        cv::Mat padded;
        int m = cv::getOptimalDFTSize(2 * input.rows);
        int n = cv::getOptimalDFTSize(2 * input.cols);
        cv::flip(input, input, 1);
        cv::copyMakeBorder(input, padded, 0, m / 2, n / 2, 0, cv::BORDER_CONSTANT, cv::Scalar::all(0));

        cv::Mat planes[2] = { cv::Mat_<float>(padded),
                              cv::Mat::zeros(padded.size(), CV_32F) };
        cv::Mat complexImage;
        cv::merge(planes, 2, complexImage);

        complexImage.convertTo(output, complexImage.type());
        cv::normalize(input, input, 0, 1, cv::NORM_MINMAX);
        return cv::Size(m, n);
    }
    catch (std::exception &e) {
        std::cout << "Exception in set_optimal_holosize: " << e.what() << "\n";
        return cv::Size(0, 0);
    }
}

// -----------------------------------------------------------------------------
// holoeye_transform: General transform based on the specified type.
// -----------------------------------------------------------------------------
cv::Mat eit_hologram::holoeye_transform(cv::Mat& padded,
                                        cv::Mat& adft_data,
                                        std::string holotype) {
    // Ensure the input has two channels.
    assert(padded.channels() == 2);
    try {
        if (holotype == "FRESNEL")
            return holoeye_fresnel(padded, adft_data);
        else if (holotype == "FRFT")
            return holoeye_fractional_ft(padded, adft_data, frft_angle);
        else if (holotype == "RPN")
            return holoeye_rpn(padded, adft_data);
        else if (holotype == "FFT")
            return holoeye_dft(padded, adft_data);
        else if (holotype == "FRECONV")
            return holoeye_convolution_extended(padded, adft_data);
        else if (holotype == "AS")
            return holoeye_angular_spectrum(padded, adft_data);
        else
            return holoeye_fresnel(padded, adft_data);
    }
    catch (std::exception &e) {
        std::cout << "Exception in holoeye_transform: " << e.what() << "\n";
        return cv::Mat();
    }
}

// -----------------------------------------------------------------------------
// holoeye_fractional_ft: Implements a fractional Fourier transform.
// -----------------------------------------------------------------------------
cv::Mat eit_hologram::holoeye_fractional_ft(cv::Mat& input,
                                            cv::Mat& cplx,
                                            float angle) {
    try {
        // Prepare two channels from the input.
        cv::Mat inputdat[2] = { cv::Mat::zeros(input.size(), input.type()),
                                cv::Mat::zeros(input.size(), input.type()) };
        cv::normalize(input, input, 0, 1, cv::NORM_MINMAX);
        cv::split(input, inputdat);

        // First chirp multiplication.
        for (int row = 0; row < input.rows; row++) {
            for (int col = 0; col < input.cols; col++) {
                float& in_re = inputdat[0].at<float>(row, col);
                float& in_im = inputdat[1].at<float>(row, col);
                std::complex<float> indat(in_re, in_im);

                // Multiply by a chirp function.
                indat *= std::exp(constants::_I * ((row * row * constants::holoeye_delta_x * constants::holoeye_delta_x +
                                                      col * col * constants::holoeye_delta_y * constants::holoeye_delta_y) / 2.0f)
                                  * std::tan(constants::pi / 2.0f - angle));

                // Scaling factors (avoid division by zero)
                float f_x = 1.0f / (((row == 0) ? 1.0f : row) * constants::holoeye_delta_x);
                float f_y = 1.0f / (((col == 0) ? 1.0f : col) * constants::holoeye_delta_y);

                indat *= std::exp(constants::_I * (row * constants::holoeye_delta_x + col * constants::holoeye_delta_y)
                                  * (f_x + f_y) / std::sin(angle));
                in_re = indat.real();
                in_im = indat.imag();
            }
        }

        cv::merge(inputdat, 2, input);
        cv::dft(input, input);
        cv::split(input, inputdat);

        // Second chirp multiplication after DFT.
        for (int row = 0; row < input.rows; row++) {
            for (int col = 0; col < input.cols; col++) {
                float f_x = 1.0f / (((row == 0) ? 1.0f : row) * constants::holoeye_delta_x);
                float f_y = 1.0f / (((col == 0) ? 1.0f : col) * constants::holoeye_delta_y);

                std::complex<float> h_a = std::exp(constants::_I * ((f_x + f_y) / 2.0f) * std::tan(constants::pi / 2.0f - angle))
                                          * std::sqrt((1.0f - constants::_I * std::tan(constants::pi / 2.0f - angle)) / (2.0f * constants::pi));

                float& in_re = inputdat[0].at<float>(row, col);
                float& in_im = inputdat[1].at<float>(row, col);
                std::complex<float> indat(in_re, in_im);
                indat *= h_a;
                in_re = indat.real();
                in_im = indat.imag();
            }
        }

        cv::merge(inputdat, 2, input);
        cv::merge(inputdat, 2, cplx);
        return input;
    }
    catch (std::exception& e) {
        std::cout << "Exception in holoeye_fractional_ft(): " << e.what() << "\n";
        return cv::Mat();
    }
}

// -----------------------------------------------------------------------------
// holoeye_angular_spectrum: Implements the angular spectrum method.
// -----------------------------------------------------------------------------
cv::Mat eit_hologram::holoeye_angular_spectrum(cv::Mat& input,
                                               cv::Mat& cplx) {
    try {
        cv::Mat inputdat[2] = { cv::Mat::zeros(input.size(), input.type()),
                                cv::Mat::zeros(input.size(), input.type()) };
        int rowmid = input.rows / 2 + 1;
        int colmid = input.cols / 2 + 1;

        cv::normalize(input, input, 0, 1, cv::NORM_MINMAX);
        cv::dft(input, input);
        cv::split(input, inputdat);

        for (int row = 0; row < input.rows; row++) {
            for (int col = 0; col < input.cols; col++) {
                float delta_x = constants::holoeye_delta_x;
                float delta_y = constants::holoeye_delta_y;
                int relRow = row - rowmid;
                int relCol = col - colmid;
                float denom_x = (relRow == 0 ? 1.0f : static_cast<float>(std::abs(relRow)));
                float denom_y = (relCol == 0 ? 1.0f : static_cast<float>(std::abs(relCol)));
                float f_x = 1.0f / (denom_x * delta_x);
                float f_y = 1.0f / (denom_y * delta_y);

                std::complex<float> h_a = std::exp(constants::_I * glob_distance *
                    std::sqrt((2.0f * constants::pi / wavelength) * (2.0f * constants::pi / wavelength)
                              - 4.0f * constants::pi * constants::pi * (f_x * f_x + f_y * f_y)));
                h_a *= std::exp(-constants::neg_I * 2.0f * constants::pi * (relRow * f_x + relCol * f_y));

                float& in_re = inputdat[0].at<float>(row, col);
                float& in_im = inputdat[1].at<float>(row, col);
                std::complex<float> indat(in_re, in_im);
                indat *= h_a;
                in_re = indat.real();
                in_im = indat.imag();
            }
        }

        cv::merge(inputdat, 2, input);
        cv::merge(inputdat, 2, cplx);
        return input;
    }
    catch (std::exception& e) {
        std::cout << "Exception in holoeye_angular_spectrum: " << e.what() << "\n";
        return cv::Mat();
    }
}

// -----------------------------------------------------------------------------
// holoeye_fresnel: Implements a Fresnel transform using an added chirp.
// -----------------------------------------------------------------------------
cv::Mat eit_hologram::holoeye_fresnel(cv::Mat& input,
                                      cv::Mat& c_data,
                                      float distance,
                                      float angle) {
    try {
        float pitch = constants::holoeye_delta_x; // Assuming same pitch for rows/cols.
        cv::Mat h_times_r[2];
        cv::split(input, h_times_r);

        // Determine the effective distance.
        float effective_distance = (distance != glob_distance) ? distance : glob_distance;
        cv::Mat chirp = holoeye_chirp(h_times_r[0], effective_distance);

        // Add the chirp only to the phase channel.
        h_times_r[1] += chirp;
        cv::Mat complexHR;
        cv::merge(h_times_r, 2, complexHR);
        cv::dft(complexHR, complexHR);

        input = complexHR;
        c_data = complexHR;
        return complexHR;
    }
    catch (std::exception &e) {
        std::cout << "Exception in holoeye_fresnel(): " << e.what() << "\n";
        return cv::Mat();
    }
}

// -----------------------------------------------------------------------------
// holoeye_convolution_extended: Uses an auxiliary chirp for extended objects.
// -----------------------------------------------------------------------------
cv::Mat eit_hologram::holoeye_convolution_extended(cv::Mat& input,
                                                   cv::Mat& cplx,
                                                   float distance) {
    try {
        cv::Mat planes[2] = { cv::Mat::zeros(input.size(), CV_32F),
                              cv::Mat::zeros(input.size(), CV_32F) };

        // Determine effective distance.
        float effective_distance = (distance == glob_distance) ? distance : glob_distance;
        cv::Mat ref = holoeye_ref_wavefront_sommerfeld(input, distance);
        input *= ref;
        cv::dft(input, input);

        cv::merge(planes, 2, input);
        cv::merge(planes, 2, cplx);
        return input;
    }
    catch (std::exception &e) {
        std::cout << "Exception in holoeye_convolution_extended: " << e.what() << "\n";
        return cv::Mat();
    }
}

// -----------------------------------------------------------------------------
// holoeye_frft: Fractional FT via an alternate algorithm (not yet implemented)
// -----------------------------------------------------------------------------
cv::Mat eit_hologram::holoeye_frft(cv::Mat& input,
                                   cv::Mat& cplx,
                                   float order) {
    try {
        // FRFT implementation by Zaktas et al. (2003) would go here.
        return input;
    }
    catch (std::exception &e) {
        std::cout << "Exception in holoeye_frft: " << e.what() << "\n";
        return cv::Mat();
    }
}

// -----------------------------------------------------------------------------
// holoeye_dft: Performs a standard DFT.
// -----------------------------------------------------------------------------
cv::Mat eit_hologram::holoeye_dft(cv::Mat& g_ROI, cv::Mat& outputdata) {
    try {
        cv::dft(g_ROI, g_ROI);
        outputdata = g_ROI;
        return g_ROI;
    }
    catch (std::exception& e) {
        std::cout << "Exception in holoeye_dft: " << e.what() << "\n";
        return cv::Mat();
    }
}

// -----------------------------------------------------------------------------
// holoeye_rpn: Implements a hologram using random phase (with twin image removal)
// -----------------------------------------------------------------------------
cv::Mat eit_hologram::holoeye_rpn(cv::Mat& g_ROI, cv::Mat& outputdata) {
    try {
        cv::Mat planes[2] = { cv::Mat::zeros(g_ROI.size(), CV_32FC1),
                              cv::Mat::zeros(g_ROI.size(), CV_32FC1) };
        cv::split(g_ROI, planes);

        cv::Mat planes_idft[2] = { cv::Mat_<float>(planes[0]),
                                   cv::Mat::zeros(planes[0].size(), CV_32FC1) };
        cv::Mat complexI(g_ROI);

        cv::Scalar meanVal = cv::mean(complexI);
        float fmean = cv::sum(meanVal)[0];

        cv::Mat planes_factor[2] = { cv::Mat::zeros(g_ROI.size(), CV_32FC1),
                                     cv::Mat::zeros(g_ROI.size(), CV_32FC1) };

        for (int row = 0; row < complexI.rows; row++) {
            for (int col = 0; col < complexI.cols; col++) {
                float& pxl_re = planes_factor[0].at<float>(row, col);
                float& pxl_im = planes_factor[1].at<float>(row, col);
                std::complex<float> compVal = complexI.at<std::complex<float>>(row, col);
                std::complex<float> scaled = (1.0f / std::sqrt(1920.0f * 1080.0f)) * compVal - std::complex<float>(fmean);
                pxl_re = scaled.real();
                pxl_im = scaled.imag();
            }
        }

        cv::Mat complex_factor;
        cv::merge(planes_factor, 2, complex_factor);

        // Set up random phase generator.
        boost::mt19937 rng;
        boost::normal_distribution<> nd(0.0, 1.0);
        boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > var_nor(rng, nd);

        cv::Mat planes_gaussian[2] = { cv::Mat::zeros(g_ROI.size(), CV_32FC1),
                                       cv::Mat::zeros(g_ROI.size(), CV_32FC1) };

        // Generate random phase.
        for (int row = 0; row < complexI.rows; row++) {
            for (int col = 0; col < complexI.cols; col++) {
                float rnd_phase_rad = var_nor();
                std::complex<float> phasevalue(std::exp(constants::neg_I * rnd_phase_rad));
                planes_gaussian[0].at<float>(row, col) = phasevalue.real();
                planes_gaussian[1].at<float>(row, col) = phasevalue.imag();
            }
        }

        cv::Mat complex_gaussian;
        cv::merge(planes_gaussian, 2, complex_gaussian);

        cv::dft(complexI, complexI);
        outputdata = complexI;

        cv::split(complex_gaussian, planes_gaussian);
        cv::split(complexI, planes);

        for (int row = 0; row < complexI.rows; row++) {
            for (int col = 0; col < complexI.cols; col++) {
                std::complex<float> val(planes_factor[0].at<float>(row, col), planes_factor[1].at<float>(row, col));
                std::complex<float> val_gauss(planes_gaussian[0].at<float>(row, col), planes_gaussian[1].at<float>(row, col));
                val_gauss *= val;
                planes_gaussian[0].at<float>(row, col) = val.real();
                planes_gaussian[1].at<float>(row, col) = val.imag();
            }
        }

        cv::Mat interphase;
        cv::phase(planes_gaussian[0], planes_gaussian[1], interphase);
        cv::Mat chirp = holoeye_chirp(interphase);

        for (int row = 0; row < complexI.rows; row++) {
            for (int col = 0; col < complexI.cols; col++) {
                std::complex<float> val(planes[0].at<float>(row, col), planes[1].at<float>(row, col));
                std::complex<float> val_gauss(planes_gaussian[0].at<float>(row, col), planes_gaussian[1].at<float>(row, col));
                float phase_ = chirp.at<float>(row, col);
                std::complex<float> sinc_filt = std::polar(1.0f, phase_);
                val += val_gauss;
                val += sinc_filt;
                planes[0].at<float>(row, col) = val.real();
                planes[1].at<float>(row, col) = val.imag();
            }
        }

        cv::merge(planes, 2, outputdata);
        cv::merge(planes, 2, g_ROI);
        return outputdata;
    }
    catch (std::exception &e) {
        std::cout << "Exception in holoeye_rpn: " << e.what() << "\n";
        return cv::Mat();
    }
}

// -----------------------------------------------------------------------------
// holoeye_rpn_no_twin: Similar to holoeye_rpn but without twin-image reduction.
// -----------------------------------------------------------------------------
cv::Mat eit_hologram::holoeye_rpn_no_twin(cv::Mat g_ROI, cv::Mat& outputdata) {
    try {
        cv::Mat padded;
        int m = cv::getOptimalDFTSize(g_ROI.rows);
        int n = cv::getOptimalDFTSize(g_ROI.cols);
        cv::copyMakeBorder(g_ROI, padded, 0, m - g_ROI.rows, 0, n - g_ROI.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

        cv::Mat planes[2] = { cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32FC1) };
        cv::Mat complexI;
        cv::merge(planes, 2, complexI);

        cv::Scalar meanVal = cv::mean(complexI);
        std::complex<float> fmean = cv::sum(meanVal)[0];

        cv::Mat planes_factor[2] = { cv::Mat::zeros(padded.size(), CV_32FC1),
                                     cv::Mat::zeros(padded.size(), CV_32FC1) };

        for (int row = 0; row < complexI.rows; row++) {
            for (int col = 0; col < complexI.cols; col++) {
                float& pxl_re = planes_factor[0].at<float>(row, col);
                float& pxl_im = planes_factor[1].at<float>(row, col);
                std::complex<float> compVal = complexI.at<std::complex<float>>(row, col);
                std::complex<float> scaled = (1.0f / std::sqrt(1920.0f * 1080.0f)) * compVal - std::complex<float>(fmean);
                pxl_re = scaled.real();
                pxl_im = scaled.imag();
            }
        }

        cv::Mat complex_factor;
        cv::merge(planes_factor, 2, complex_factor);

        boost::mt19937 rng;
        boost::normal_distribution<> nd(0.0, 1.0);
        boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > var_nor(rng, nd);

        cv::Mat planes_gaussian[2] = { cv::Mat::zeros(padded.size(), CV_32FC1),
                                       cv::Mat::zeros(padded.size(), CV_32FC1) };

        for (int row = 0; row < complexI.rows; row++) {
            for (int col = 0; col < complexI.cols; col++) {
                float rnd_phase_rad = var_nor();
                std::complex<float> phasevalue(std::exp(constants::neg_I * rnd_phase_rad));
                planes_gaussian[0].at<float>(row, col) = phasevalue.real();
                planes_gaussian[1].at<float>(row, col) = phasevalue.imag();
            }
        }

        cv::Mat complex_gaussian;
        cv::merge(planes_gaussian, 2, complex_gaussian);

        cv::dft(complexI, complexI);
        outputdata = complexI;

        cv::split(complex_gaussian, planes_gaussian);
        cv::split(complexI, planes);

        for (int row = 0; row < complexI.rows; row++) {
            for (int col = 0; col < complexI.cols; col++) {
                std::complex<float> val(planes_factor[0].at<float>(row, col), planes_factor[1].at<float>(row, col));
                std::complex<float> val_gauss(planes_gaussian[0].at<float>(row, col), planes_gaussian[1].at<float>(row, col));
                val_gauss *= val;
                planes_gaussian[0].at<float>(row, col) = val.real();
                planes_gaussian[1].at<float>(row, col) = val.imag();
            }
        }

        cv::Mat interphase;
        cv::phase(planes_gaussian[0], planes_gaussian[1], interphase);
        cv::Mat sinc_filtered = holoeye_chirp(interphase);

        for (int row = 0; row < complexI.rows; row++) {
            for (int col = 0; col < complexI.cols; col++) {
                std::complex<float> val(planes[0].at<float>(row, col), planes[1].at<float>(row, col));
                std::complex<float> val_gauss(planes_gaussian[0].at<float>(row, col), planes_gaussian[1].at<float>(row, col));
                float phase_ = sinc_filtered.at<float>(row, col);
                std::complex<float> sinc_filt = std::polar(1.0f, phase_);
                val += val_gauss;
                val -= sinc_filt;
                planes[0].at<float>(row, col) = val.real();
                planes[1].at<float>(row, col) = val.imag();
            }
        }

        cv::merge(planes, 2, g_ROI);
        return g_ROI;
    }
    catch (std::exception &e) {
        std::cout << "Exception in holoeye_rpn_no_twin: " << e.what() << "\n";
        return cv::Mat();
    }
}

// -----------------------------------------------------------------------------
// qshow: Quick display function for debugging.
// -----------------------------------------------------------------------------
void eit_hologram::qshow(cv::Mat& in, std::string name) {
    if (in.channels() == 1) {
        cv::imshow("Qshow " + name, in);
    }
    else if (in.channels() == 2) {
        cv::Mat planes[2] = { cv::Mat::zeros(in.size(), CV_32F),
                              cv::Mat::zeros(in.size(), CV_32F) };
        cv::split(in, planes);
        cv::imshow("Qshow re " + name, planes[0]);
        cv::imshow("Qshow im " + name, planes[1]);
    }
}

// -----------------------------------------------------------------------------
// Inverse DFT helper functions
// -----------------------------------------------------------------------------
cv::Mat idft_one(cv::Mat inputdata) {
    inputdata.convertTo(inputdata, CV_32FC1);
    cv::Mat inverse = inputdata;
    cv::idft(inverse, inverse);
    cv::Mat planes[2] = { cv::Mat_<float>(inputdata), cv::Mat::zeros(inputdata.size(), inputdata.type()) };
    cv::split(inverse, planes);
    cv::magnitude(planes[0], planes[1], planes[0]);
    cv::Mat magI = planes[0];
    magI += cv::Scalar::all(1);
    // Optionally: exp(magI, magI);
    cv::normalize(magI, magI, 0, 1, cv::NORM_MINMAX);
    return magI;
}

cv::Mat idft_two(cv::Mat magI) {
    magI.convertTo(magI, CV_32FC1);
    cv::dft(magI, magI, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT);
    cv::normalize(magI, magI, 0, 1, cv::NORM_MINMAX);
    return magI;
}

} // namespace eit
