//
// Created by piotro on 15.02.25.
//
// File: ./test/optimize_test.cpp

#include "gtest/gtest.h"
#include "../include/optimize.hpp"
#include <opencv2/opencv.hpp>
#include <stdexcept>

using namespace cv;
using namespace std;
using namespace eit;

// Test that basic_operation computes the expected pixel differences.
// Note: basic_operation processes only for column indices > 1 (i.e. starting with the third element in each row).
TEST(BasicOperationTest, CorrectComputation) {
    // Create three 2x3 single-channel matrices of type CV_8U.
    // Row-wise values:
    // input:       [10, 20, 30]
    //              [40, 50, 60]
    // input_before:[ 1,  2,  3]
    //              [ 4,  5,  6]
    // desired:     [ 4,  5,  6]
    //              [ 7,  8,  9]
    Mat input       = (Mat_<uchar>(2, 3) << 10, 20, 30, 40, 50, 60);
    Mat input_before= (Mat_<uchar>(2, 3) <<  1,  2,  3,  4,  5,  6);
    Mat desired     = (Mat_<uchar>(2, 3) <<  4,  5,  6,  7,  8,  9);

    // For columns 0 and 1, basic_operation leaves the values unchanged.
    // For column 2, it computes:
    //   diff = input - (input_before - desired)
    // For row 0: 30 - (3 - 6) = 30 - (-3) = 33.
    // For row 1: 60 - (6 - 9) = 60 - (-3) = 63.
    // Thus, expected output:
    // Row 0: [10, 20, 33]
    // Row 1: [40, 50, 63]
    Mat expected = (Mat_<uchar>(2, 3) << 10, 20, 33, 40, 50, 63);

    Mat output = basic_operation(input, input_before, desired);

    // Verify that the computed output exactly matches the expected matrix.
    EXPECT_EQ(countNonZero(output != expected), 0);
}

// Test that mismatched sizes throw an invalid_argument exception.
TEST(BasicOperationTest, MismatchedSizeThrows) {
    Mat input       = Mat::ones(2, 3, CV_8U);
    Mat input_before= Mat::ones(3, 3, CV_8U); // Different size.
    Mat desired     = Mat::ones(2, 3, CV_8U);

    EXPECT_THROW(basic_operation(input, input_before, desired), std::invalid_argument);
}

// Test that mismatched types throw an invalid_argument exception.
TEST(BasicOperationTest, MismatchedTypeThrows) {
    Mat input       = Mat::ones(2, 3, CV_8U);
    Mat input_before= Mat::ones(2, 3, CV_8U);
    Mat desired     = Mat::ones(2, 3, CV_32F); // Different type.

    EXPECT_THROW(basic_operation(input, input_before, desired), std::invalid_argument);
}
