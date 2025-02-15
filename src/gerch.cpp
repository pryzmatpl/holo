#include "../include/gerch.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cmath>
#include <stdexcept>

namespace eit {

//------------------------------------------------------------------------------
// Constructor & Destructor
//------------------------------------------------------------------------------
gerch::gerch() {
    // Initialization if needed.
}

gerch::~gerch() {
    // Cleanup if needed.
}

//------------------------------------------------------------------------------
// error_less_than: Placeholder for error checking (not yet implemented).
//------------------------------------------------------------------------------
bool gerch::error_less_than(cv::Mat& in, cv::Mat& initial) {
    // TODO: Implement error measurement between 'in' and 'initial'.
    return false;
}

//------------------------------------------------------------------------------
// Helper function: amphasetoreim
// Converts an amplitude-phase representation (in two channels)
// into a real-imaginary representation.
// Assumes that input[0] holds the amplitude and input[1] holds the phase.
//------------------------------------------------------------------------------
void amphasetoreim(cv::Mat input[2]) {
    for (int row = 0; row < input[0].rows; row++) {
        for (int col = 0; col < input[0].cols; col++) {
            // 're' is amplitude and 'im' is phase.
            float &ampl = input[0].at<float>(row, col);
            float &phase = input[1].at<float>(row, col);
            // Convert to real-imaginary representation:
            float realPart = ampl * std::cos(phase);
            float imagPart = ampl * std::sin(phase);
            ampl = realPart;
            phase = imagPart;
        }
    }
}

//------------------------------------------------------------------------------
// Helper function: reimtoamphase
// Converts a real-imaginary representation (in two channels)
// into an amplitude-phase representation.
//------------------------------------------------------------------------------
void reimtoamphase(cv::Mat input[2]) {
    cv::Mat ampl, phases;
    cv::phase(input[0], input[1], phases);
    cv::magnitude(input[0], input[1], ampl);
    input[0] = ampl;
    input[1] = phases;
}

//------------------------------------------------------------------------------
// Operator() overload: Implements the Gerchberg-Saxton (GS) algorithm.
//------------------------------------------------------------------------------
void gerch::operator()(cv::Mat& input, cv::Mat& output) {
    std::cout << "Going into GS algorithm" << std::endl;

    // Create two-channel (real-imaginary) representation for the complex image.
    cv::Mat input_cplx[2] = {
        cv::Mat::zeros(input.size(), CV_32F),
        cv::Mat::zeros(input.size(), CV_32F)
    };

    // Call typetest (assumed defined in fft.cpp within namespace eit)
    typetest(input);
    typetest(input_cplx[0]);

    // Create matrices for a uniform amplitude (all ones) and phase.
    cv::Mat intensity_sqrt = cv::Mat::zeros(input.size(), CV_32F);
    cv::Mat singles = cv::Mat::zeros(input.size(), CV_32F);
    cv::Mat phase_temp = cv::Mat::zeros(input.size(), CV_32F);

    for (int row = 0; row < input.rows; row++) {
        for (int col = 0; col < input.cols; col++) {
            intensity_sqrt.at<float>(row, col) = 1.0f;
            singles.at<float>(row, col) = 1.0f;
        }
    }

    cv::Mat first_iter(input.size(), CV_32FC1);
    typetest(first_iter);
    typetest(input_cplx[0]);
    typetest(input_cplx[1]);

    // Initialize the phase channel with a random normal distribution.
    cv::randn(input_cplx[1], 128, 32);

    // Iteratively update the estimate using the Gerchberg-Saxton algorithm.
    for (int a = 0; a < 100000; a++) {
        amphasetoreim(input_cplx);                      // Convert amplitude-phase to real-imaginary.
        cv::merge(input_cplx, 2, first_iter);           // Merge channels into a complex image.
        cv::dft(first_iter, first_iter);                // Forward DFT.
        cv::split(first_iter, input_cplx);              // Split back into channels.

        reimtoamphase(input_cplx);                      // Convert back to amplitude-phase.
        intensity_sqrt.copyTo(input_cplx[0]);           // Force the amplitude to be uniform (all ones).

        amphasetoreim(input_cplx);                      // Convert to real-imaginary again.
        cv::merge(input_cplx, 2, first_iter);           // Merge channels.
        cv::dft(first_iter, first_iter, cv::DFT_INVERSE);// Inverse DFT.
        cv::split(first_iter, input_cplx);              // Split channels.

        reimtoamphase(input_cplx);                      // Convert to amplitude-phase.
        singles.copyTo(input_cplx[0]);                  // Replace amplitude with uniform ones.
    }

    cv::merge(input_cplx, 2, output);
}

//------------------------------------------------------------------------------
// eit_lsq: Solves a least-squares problem using OpenCV's solve function.
//------------------------------------------------------------------------------
cv::Mat gerch::eit_lsq(cv::Mat& in, cv::Mat& cmp) {
    cv::Mat returnee;
    if (!cv::solve(in, cmp, returnee, cv::DECOMP_NORMAL)) {
        throw std::runtime_error("Not possible to solve the problem");
    }
    return returnee;
}

} // namespace eit
