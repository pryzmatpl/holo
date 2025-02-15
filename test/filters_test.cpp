//
// Created by piotro on 15.02.25.
//
// File: ./test/filters_test.cpp

#include "gtest/gtest.h"

#include "../include/filters.hpp"
#include "../include/consts.hpp"
#include "../include/fft.hpp"  // For typetest(), if needed.
#include <opencv2/opencv.hpp>
#include <boost/math/special_functions/sinc.hpp>
#include <cmath>

using namespace cv;
using namespace std;
using namespace eit;

//------------------------------------------------------------------------------
// Test: holoeye_filter_c1 with a custom lambda (doubling each element)
//------------------------------------------------------------------------------
TEST(FiltersTest, HoloeyeFilterC1_DoubleLambda) {
    // Create a simple 2x2 single-channel matrix.
    Mat input = Mat::ones(2, 2, CV_32F);
    // Define a lambda that multiplies every element by 2.
    auto doubleLambda = [](Mat &m) {
        m.forEach<float>([](float &p, const int* /*pos*/) {
            p *= 2.0f;
        });
    };
    // Instantiate an object of eit_hologram to call its member function.
    eit_hologram holo;
    Mat output = holo.holoeye_filter_c1(input, doubleLambda);
    // Each element should now equal 2.
    for (int i = 0; i < output.rows; ++i)
        for (int j = 0; j < output.cols; ++j)
            EXPECT_NEAR(output.at<float>(i, j), 2.0f, 1e-5);
}

//------------------------------------------------------------------------------
// Test: filter_remavg lambda
// Subtracting the mean from each element should yield (value - mean).
TEST(FiltersTest, FilterRemavg) {
    // Create a 3x3 matrix with known values.
    Mat input = (Mat_<float>(3,3) << 1, 2, 3, 4, 5, 6, 7, 8, 9);
    Scalar meanVal = mean(input);
    float meanF = static_cast<float>(meanVal[0]);

    Mat filtered = input.clone();
    filter_remavg(filtered);

    // Check each element equals (original - mean).
    for (int i = 0; i < input.rows; ++i)
        for (int j = 0; j < input.cols; ++j)
            EXPECT_NEAR(filtered.at<float>(i, j), input.at<float>(i, j) - meanF, 1e-5);
}

//------------------------------------------------------------------------------
// Test: holoeye_filter (using "REMAVG" option)
// Should preserve size and type.
TEST(FiltersTest, HoloeyeFilter_Remavg) {
    Mat input = (Mat_<float>(3,3) << 10, 20, 30, 40, 50, 60, 70, 80, 90);
    eit_hologram holo;
    Mat output = holo.holoeye_filter(input, "REMAVG");
    EXPECT_EQ(output.size(), input.size());
    EXPECT_EQ(output.type(), input.type());
}

//------------------------------------------------------------------------------
// Test: holoeye_filter_spatial returns a single-channel float image
// with the same size as the input.
TEST(FiltersTest, HoloeyeFilterSpatial) {
    Mat input = Mat::ones(4, 4, CV_32F);
    Mat spatial_filter = Mat::ones(4, 4, CV_32F);
    eit_hologram holo;
    Mat output = holo.holoeye_filter_spatial(input, spatial_filter);
    EXPECT_EQ(output.size(), input.size());
    EXPECT_EQ(output.type(), CV_32F);
}

//------------------------------------------------------------------------------
// Test: filter_sinc preserves size and type for CV_32FC3 input.
//------------------------------------------------------------------------------
TEST(FiltersTest, FilterSinc_TypePreservation) {
    Mat input(4, 4, CV_32FC3, Scalar(1.0f, 0.0f, 0.0f));
    Mat output = input.clone();
    filter_sinc(output);
    EXPECT_EQ(output.size(), input.size());
    EXPECT_EQ(output.type(), input.type());
}

//------------------------------------------------------------------------------
// Test: filter_rpn_ocv modifies the input.
//------------------------------------------------------------------------------
TEST(FiltersTest, FilterRPNOcv_Modification) {
    Mat input = Mat::ones(4, 4, CV_32F) * 50;
    Mat output = input.clone();
    filter_rpn_ocv(output);
    // Expect that at least one pixel value is different.
    EXPECT_NE(countNonZero(output != input), 0);
}

//------------------------------------------------------------------------------
// Test: filter_spherical preserves size and type.
TEST(FiltersTest, FilterSpherical_SizeType) {
    Mat input = Mat::ones(5, 5, CV_32F) * 100;
    Mat output = input.clone();
    filter_spherical(output);
    EXPECT_EQ(output.size(), input.size());
    EXPECT_EQ(output.type(), input.type());
}

//------------------------------------------------------------------------------
// Test: filter_linear_phase_load preserves size and type.
TEST(FiltersTest, FilterLinearPhaseLoad_SizeType) {
    Mat input(5, 5, CV_32FC2, Scalar(1.0f, 0.0f));
    Mat output = input.clone();
    filter_linear_phase_load(output);
    EXPECT_EQ(output.size(), input.size());
    EXPECT_EQ(output.type(), input.type());
}

//------------------------------------------------------------------------------
// Test: filter_real_laplacian preserves size and type.
TEST(FiltersTest, FilterRealLaplacian_SizeType) {
    Mat input;
    randu(input, Scalar(0), Scalar(255));
    input.convertTo(input, CV_32F);
    Mat output = input.clone();
    filter_real_laplacian(output);
    EXPECT_EQ(output.size(), input.size());
    EXPECT_EQ(output.type(), input.type());
}

//------------------------------------------------------------------------------
// Test: filter_clear_center preserves size and type.
TEST(FiltersTest, FilterClearCenter_SizeType) {
    Mat input(10, 10, CV_32FC2, Scalar(5.0f, 5.0f));
    Mat output = input.clone();
    filter_clear_center(output);
    EXPECT_EQ(output.size(), input.size());
    EXPECT_EQ(output.type(), input.type());
}
