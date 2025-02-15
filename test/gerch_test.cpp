//
// Created by piotro on 15.02.25.
//
// File: ./test/gerch_test.cpp

#include "gtest/gtest.h"
#include "../include/gerch.hpp"
#include <opencv2/opencv.hpp>
#include <cmath>
#include <stdexcept>

using namespace cv;
using namespace std;
using namespace eit;

//------------------------------------------------------------------------------
// Test: error_less_than should return false (since it's a placeholder).
//------------------------------------------------------------------------------
TEST(GerchTest, ErrorLessThanReturnsFalse) {
    Mat in = Mat::ones(3, 3, CV_32F);
    Mat initial = Mat::zeros(3, 3, CV_32F);
    gerch gs;
    EXPECT_FALSE(gs.error_less_than(in, initial));
}

//------------------------------------------------------------------------------
// Test: amphasetoreim converts amplitude-phase to real-imaginary representation.
// For a known amplitude and phase, verify the conversion.
TEST(GerchTest, AmphasetoreimConversion) {
    // Create a 1x1 matrix for amplitude and phase.
    Mat ampl = (Mat_<float>(1, 1) << 2.0f);
    Mat phase = (Mat_<float>(1, 1) << static_cast<float>(M_PI/4)); // 45 degrees

    // Pack into an array.
    Mat channels[2] = { ampl.clone(), phase.clone() };

    // Call conversion function.
    amphasetoreim(channels);

    // Expected: real = 2*cos(pi/4) = 2/sqrt2, imag = 2*sin(pi/4) = 2/sqrt2.
    float expected = 2.0f / std::sqrt(2.0f);
    EXPECT_NEAR(channels[0].at<float>(0, 0), expected, 1e-5);
    EXPECT_NEAR(channels[1].at<float>(0, 0), expected, 1e-5);
}

//------------------------------------------------------------------------------
// Test: reimtoamphase converts back from real-imaginary to amplitude-phase.
// Using the result from the previous test should recover the original amplitude and phase.
TEST(GerchTest, ReimtoamphaseConversion) {
    // Start with known real and imaginary parts.
    float realVal = 2.0f / std::sqrt(2.0f);
    float imagVal = 2.0f / std::sqrt(2.0f);
    Mat realMat = (Mat_<float>(1, 1) << realVal);
    Mat imagMat = (Mat_<float>(1, 1) << imagVal);
    Mat channels[2] = { realMat.clone(), imagMat.clone() };

    // Call conversion function.
    reimtoamphase(channels);

    // Expected amplitude = 2.0, expected phase = pi/4.
    EXPECT_NEAR(channels[0].at<float>(0, 0), 2.0f, 1e-5);
    EXPECT_NEAR(channels[1].at<float>(0, 0), static_cast<float>(M_PI/4), 1e-5);
}

//------------------------------------------------------------------------------
// Test: operator() of Gerch (Gerchberg-Saxton algorithm)
// Since the algorithm iterates 100000 times, we test on a very small input and
// verify that output has the correct size and type.
//------------------------------------------------------------------------------
TEST(GerchTest, GSOperatorOutputSizeAndType) {
    // Create a 2x2 single-channel matrix (CV_32F) as input.
    Mat input = Mat::ones(2, 2, CV_32F);
    Mat output;
    gerch gs;

    // Run the GS algorithm. (Note: This will run the full iteration loop.)
    // For a 2x2 matrix, even 100000 iterations should complete in a reasonable time.
    gs(input, output);

    // The output is produced by merging two CV_32F channels.
    EXPECT_EQ(output.size(), input.size());
    EXPECT_EQ(output.type(), CV_32FC2);
}

//------------------------------------------------------------------------------
// Test: eit_lsq solves a simple linear system correctly.
// Example: Solve [1 2; 3 4] * x = [5; 11], solution x = [1; 2].
//------------------------------------------------------------------------------
TEST(GerchTest, LSQTest) {
    Mat A = (Mat_<float>(2,2) << 1, 2, 3, 4);
    Mat b = (Mat_<float>(2,1) << 5, 11);
    gerch gs;
    Mat x = gs.eit_lsq(A, b);

    // Expected solution: [1; 2].
    EXPECT_NEAR(x.at<float>(0, 0), 1.0f, 1e-4);
    EXPECT_NEAR(x.at<float>(1, 0), 2.0f, 1e-4);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
