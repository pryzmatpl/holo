//
// Created by piotro on 15.02.25.
//
// File: ./test/fft_test.cpp

#include "gtest/gtest.h"
#include "../include/fft.hpp"
#include "../include/consts.hpp"
#include <opencv2/opencv.hpp>
#include <cmath>
#include <complex>

using namespace cv;
using namespace std;
using namespace eit;

// Define external globals required by fft.cpp.
extern float wavelength;
extern float glob_distance;
extern float frft_angle;

//------------------------------------------------------------------------------
// Test: typetest
// Simply ensure that calling typetest() does not throw any exceptions.
//------------------------------------------------------------------------------
TEST(FFTTest, TypetestDoesNotThrow) {
    Mat testMat = Mat::ones(5, 5, CV_32F);
    EXPECT_NO_THROW(typetest(testMat));
}

//------------------------------------------------------------------------------
// Test: tile_to_fhd
// Create a small test image and ensure that the output has the expected size and type.
//------------------------------------------------------------------------------
TEST(FFTTest, TileToFHDSizeAndType) {
    // Create a simple grayscale image.
    Mat input = Mat::ones(10, 10, CV_32F);
    // Create an instance to call non-static member function.
    eit_hologram holo;
    Mat output = holo.tile_to_fhd(input);
    int expectedCols = static_cast<int>(constants::holoeye_width);
    int expectedRows = static_cast<int>(constants::holoeye_height);
    EXPECT_EQ(output.cols, expectedCols);
    EXPECT_EQ(output.rows, expectedRows);
    EXPECT_EQ(output.type(), input.type());
}

//------------------------------------------------------------------------------
// Test: tile_to_fhd_amp
//------------------------------------------------------------------------------
TEST(FFTTest, TileToFHDAmpSizeAndType) {
    Mat input = Mat::ones(10, 10, CV_32F);
    eit_hologram holo;
    Mat output = holo.tile_to_fhd_amp(input);
    int expectedCols = static_cast<int>(constants::holoeye_width);
    int expectedRows = static_cast<int>(constants::holoeye_height);
    EXPECT_EQ(output.cols, expectedCols);
    EXPECT_EQ(output.rows, expectedRows);
    EXPECT_EQ(output.type(), input.type());
}

//------------------------------------------------------------------------------
// Test: set_optimal_holosize
// Ensure that an input image yields a non-zero optimal size and non-empty output.
TEST(FFTTest, SetOptimalHoloSize) {
    Mat input = Mat::ones(20, 20, CV_32F);
    Mat output;
    eit_hologram holo;
    Size optimalSize = holo.set_optimal_holosize(input, output);
    EXPECT_GT(optimalSize.width, 0);
    EXPECT_GT(optimalSize.height, 0);
    EXPECT_FALSE(output.empty());
}

//------------------------------------------------------------------------------
// Test: holoeye_transform using "FFT" type.
//------------------------------------------------------------------------------
TEST(FFTTest, HoloeyeTransformFFT) {
    // Create a dummy two-channel (complex) image.
    Mat realPart = Mat::ones(10, 10, CV_32F);
    Mat imagPart = Mat::zeros(10, 10, CV_32F);
    Mat input;
    merge(vector<Mat>{realPart, imagPart}, input);
    Mat adft_data;
    eit_hologram holo;
    // Passing "FFT" as holotype.
    Mat output = holo.holoeye_transform(input, adft_data, "FFT");
    EXPECT_FALSE(output.empty());
    EXPECT_EQ(output.size(), input.size());
    EXPECT_EQ(output.channels(), 2);
}

//------------------------------------------------------------------------------
// Test: holoeye_fractional_ft
//------------------------------------------------------------------------------
TEST(FFTTest, HoloeyeFractionalFT) {
    Mat realPart = Mat::ones(8, 8, CV_32F);
    Mat imagPart = Mat::zeros(8, 8, CV_32F);
    Mat input;
    merge(vector<Mat>{realPart, imagPart}, input);
    Mat cplx;
    eit_hologram holo;
    Mat output = holo.holoeye_fractional_ft(input, cplx, frft_angle);
    EXPECT_FALSE(output.empty());
    EXPECT_EQ(output.size(), input.size());
    EXPECT_EQ(output.channels(), 2);
    EXPECT_EQ(cplx.size(), input.size());
    EXPECT_EQ(cplx.channels(), 2);
}

//------------------------------------------------------------------------------
// Test: holoeye_angular_spectrum
//------------------------------------------------------------------------------
TEST(FFTTest, HoloeyeAngularSpectrum) {
    Mat realPart = Mat::ones(8, 8, CV_32F);
    Mat imagPart = Mat::zeros(8, 8, CV_32F);
    Mat input;
    merge(vector<Mat>{realPart, imagPart}, input);
    Mat cplx;
    eit_hologram holo;
    Mat output = holo.holoeye_angular_spectrum(input, cplx);
    EXPECT_FALSE(output.empty());
    EXPECT_EQ(output.size(), input.size());
    EXPECT_EQ(output.channels(), 2);
}

//------------------------------------------------------------------------------
// Test: holoeye_fresnel
//------------------------------------------------------------------------------
TEST(FFTTest, HoloeyeFresnel) {
    Mat realPart = Mat::ones(8, 8, CV_32F);
    Mat imagPart = Mat::zeros(8, 8, CV_32F);
    Mat input;
    merge(vector<Mat>{realPart, imagPart}, input);
    Mat c_data;
    eit_hologram holo;
    Mat output = holo.holoeye_fresnel(input, c_data, 1.0f, 0.0f);
    EXPECT_FALSE(output.empty());
    EXPECT_EQ(output.size(), input.size());
    EXPECT_EQ(output.channels(), 2);
    EXPECT_EQ(c_data.size(), input.size());
}

//------------------------------------------------------------------------------
// Test: holoeye_dft
//------------------------------------------------------------------------------
TEST(FFTTest, HoloeyeDFTSizeAndType) {
    Mat input = Mat::ones(8, 8, CV_32F);
    Mat outputdata;
    eit_hologram holo;
    Mat output = holo.holoeye_dft(input, outputdata);
    EXPECT_EQ(output.size(), input.size());
    EXPECT_EQ(output.type(), input.type());
}

//------------------------------------------------------------------------------
// Test: idft_one normalization
//------------------------------------------------------------------------------
TEST(FFTTest, IDFTOneNormalization) {
    Mat input = Mat::ones(8, 8, CV_32F);
    Mat dftInput;
    dft(input, dftInput);
    // Call free function idft_one from the namespace.
    Mat output = eit::idft_one(dftInput);
    double minVal, maxVal;
    minMaxLoc(output, &minVal, &maxVal);
    EXPECT_GE(minVal, 0.0);
    EXPECT_LE(maxVal, 1.0);
}

//------------------------------------------------------------------------------
// Test: idft_two normalization
//------------------------------------------------------------------------------
TEST(FFTTest, IDFTTwoNormalization) {
    Mat input = Mat::ones(8, 8, CV_32F);
    Mat dftInput;
    dft(input, dftInput, DFT_COMPLEX_OUTPUT);
    Mat output = eit::idft_two(dftInput);
    double minVal, maxVal;
    minMaxLoc(output, &minVal, &maxVal);
    EXPECT_GE(minVal, 0.0);
    EXPECT_LE(maxVal, 1.0);
}

//------------------------------------------------------------------------------
// Test: holoeye_rpn
//------------------------------------------------------------------------------
TEST(FFTTest, HoloeyeRPNType) {
    Mat input = Mat::ones(8, 8, CV_32FC2);
    Mat outputdata;
    eit_hologram holo;
    Mat output = holo.holoeye_rpn(input, outputdata);
    EXPECT_EQ(output.size(), input.size());
    EXPECT_EQ(output.type(), CV_32FC2);
}

//------------------------------------------------------------------------------
// Test: holoeye_rpn_no_twin
//------------------------------------------------------------------------------
TEST(FFTTest, HoloeyeRPNNoTwinType) {
    Mat input = Mat::ones(8, 8, CV_32FC2);
    Mat outputdata;
    eit_hologram holo;
    Mat output = holo.holoeye_rpn_no_twin(input, outputdata);
    EXPECT_EQ(output.size(), input.size());
    EXPECT_EQ(output.type(), CV_32FC2);
}
