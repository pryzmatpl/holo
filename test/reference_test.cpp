//
// Created by piotro on 15.02.25.
// File: ./test/reference_test.cpp

#include "gtest/gtest.h"

#include "../include/fft.hpp"      // This header includes definitions for eit_hologram
#include "../include/consts.hpp"   // Contains constants::pi, etc.
#include <opencv2/opencv.hpp>
#include <complex>
#include <cmath>

// For convenience
using namespace cv;
using namespace std;
using namespace eit;

// Set a default global distance used in reference functions.
float glob_distance = 1.0f;

//------------------------------------------------------------------------------
// Test for holoeye_movavg:
// For a constant input, the inner pixels' difference from their 3x3 average should be zero.
//------------------------------------------------------------------------------
TEST(HoloeyeMovAvgTest, ConstantInputReturnsZero) {
    Mat input = Mat::ones(5, 5, CV_32F) * 5.0f;  // 5x5 matrix, all values 5.0
    Mat output = eit_hologram::holoeye_movavg(input);

    // For inner pixels (rows 1 to 3, cols 1 to 3), expected difference is 0.
    for (int row = 1; row < 4; ++row) {
        for (int col = 1; col < 4; ++col) {
            EXPECT_NEAR(output.at<float>(row, col), 0.0f, 1e-5);
        }
    }
}

//------------------------------------------------------------------------------
// Test for holoeye_ref_wavefront_flat_phase:
// For a constant 2-channel input with amplitude 1 and phase 0, applying a flat phase shift
// should result in every pixel being multiplied by e^(i*angle).
//------------------------------------------------------------------------------
TEST(HoloeyeRefWavefrontFlatPhaseTest, AppliesConstantPhaseShift) {
    // Create a 3x3 2-channel float matrix: amplitude = 1, phase = 0.
    Mat input(3, 3, CV_32FC2, Scalar(1.0f, 0.0f));
    float angle = M_PI / 4;  // 45 degrees phase shift

    // Apply flat phase reference.
    eit_hologram::holoeye_ref_wavefront_flat_phase(input, angle);

    // Expected: each pixel equals (cos(angle), sin(angle)).
    float expectedRe = cos(angle);
    float expectedIm = sin(angle);
    for (int row = 0; row < input.rows; ++row) {
        for (int col = 0; col < input.cols; ++col) {
            Vec2f val = input.at<Vec2f>(row, col);
            EXPECT_NEAR(val[0], expectedRe, 1e-5);
            EXPECT_NEAR(val[1], expectedIm, 1e-5);
        }
    }
}

//------------------------------------------------------------------------------
// Test for holoeye_ref_wavefront_phase2:
// The function returns a single-channel phase image whose values should be in [-pi, pi].
//------------------------------------------------------------------------------
TEST(HoloeyeRefWavefrontPhase2Test, ReturnsPhaseImageInRange) {
    Mat input = Mat::ones(4, 4, CV_32F) * 2.0f;  // Arbitrary constant image.
    Mat output = eit_hologram::holoeye_ref_wavefront_phase2(input);

    EXPECT_EQ(output.channels(), 1);
    EXPECT_EQ(output.size(), input.size());

    // Verify that each pixel's phase is within [-pi, pi].
    for (int row = 0; row < output.rows; ++row) {
        for (int col = 0; col < output.cols; ++col) {
            float phase = output.at<float>(row, col);
            EXPECT_GE(phase, -constants::pi);
            EXPECT_LE(phase, constants::pi);
        }
    }
}

//------------------------------------------------------------------------------
// Test for holoeye_chirp:
// The output should have the same size and type as the input.
//------------------------------------------------------------------------------
TEST(HoloeyeChirpTest, OutputSizeAndType) {
    // Create a simple 3x3 2-channel float matrix.
    Mat input(3, 3, CV_32FC2, Scalar(1.0f, 0.0f));
    Mat output = eit_hologram::holoeye_chirp(input, 1.0f);

    EXPECT_EQ(output.size(), input.size());
    EXPECT_EQ(output.type(), input.type());
}

//------------------------------------------------------------------------------
// Test for holoeye_reference using the "FLAT" branch:
// For a known input, the output should match the flat phase transformation.
//------------------------------------------------------------------------------
TEST(HoloeyeReferenceTest, FlatReference) {
    // Create a 3x3 2-channel matrix with amplitude 1 and phase 0.
    Mat input(3, 3, CV_32FC2, Scalar(1.0f, 0.0f));
    float angle = M_PI / 6;  // 30 degrees (arbitrary choice)

    // Manually compute expected result: every pixel becomes e^(i*angle).
    Mat expected(3, 3, CV_32FC2);
    for (int row = 0; row < expected.rows; row++) {
        for (int col = 0; col < expected.cols; col++) {
            expected.at<Vec2f>(row, col)[0] = cos(angle);
            expected.at<Vec2f>(row, col)[1] = sin(angle);
        }
    }

    // Use the "FLAT" branch of holoeye_reference.
    Mat output = eit_hologram::holoeye_reference(input, "FLAT");

    for (int row = 0; row < output.rows; row++) {
        for (int col = 0; col < output.cols; col++) {
            Vec2f outVal = output.at<Vec2f>(row, col);
            Vec2f expVal = expected.at<Vec2f>(row, col);
            EXPECT_NEAR(outVal[0], expVal[0], 1e-4);
            EXPECT_NEAR(outVal[1], expVal[1], 1e-4);
        }
    }
}

//------------------------------------------------------------------------------
// Test for holoeye_reference with an unsupported wf_type:
// The function should complete without error and return a normalized image.
//------------------------------------------------------------------------------
TEST(HoloeyeReferenceTest, UnsupportedTypeDoesNotCrash) {
    Mat input = Mat::ones(3, 3, CV_32FC1);
    Mat output = eit_hologram::holoeye_reference(input, "UNKNOWN");
    EXPECT_EQ(output.size(), input.size());
}

//------------------------------------------------------------------------------
// Test for holoeye_ref_wavefront_squareaperture:
// The output should have the same size and type as the input.
TEST(HoloeyeRefWavefrontSquareApertureTest, OutputSizeAndType) {
    Mat input(4, 4, CV_32FC2, Scalar(1.0f, 0.0f));
    Mat output = eit_hologram::holoeye_ref_wavefront_squareaperture(input, 1.0f);
    EXPECT_EQ(output.size(), input.size());
    EXPECT_EQ(output.type(), input.type());
}

//------------------------------------------------------------------------------
// Test for holoeye_ref_wavefront_focal:
// The function should preserve the input size and type.
TEST(HoloeyeRefWavefrontFocalTest, OutputSizeAndType) {
    Mat input(4, 4, CV_32FC2, Scalar(1.0f, 0.0f));
    eit_hologram::holoeye_ref_wavefront_focal(input);
    EXPECT_EQ(input.size(), Size(4, 4));
    EXPECT_EQ(input.type(), CV_32FC2);
}

//------------------------------------------------------------------------------
// Test for holoeye_ref_wavefront_sommerfeld:
// The output should have the same size and type as the input.
TEST(HoloeyeRefWavefrontSommerfeldTest, OutputSizeAndType) {
    Mat input(4, 4, CV_32FC2, Scalar(1.0f, 0.0f));
    Mat output = eit_hologram::holoeye_ref_wavefront_sommerfeld(input, 1.0f);
    EXPECT_EQ(output.size(), input.size());
    EXPECT_EQ(output.type(), input.type());
}

//------------------------------------------------------------------------------
// Test for holoeye_ref_wavefront_phase:
// The output should have the same size and type as the input.
TEST(HoloeyeRefWavefrontPhaseTest, OutputSizeAndType) {
    Mat input(4, 4, CV_32FC2, Scalar(1.0f, 0.0f));
    Mat output = eit_hologram::holoeye_ref_wavefront_phase(input);
    EXPECT_EQ(output.size(), input.size());
    EXPECT_EQ(output.type(), CV_32FC2);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

