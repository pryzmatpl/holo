//
// Created by piotro on 15.02.25.
//

#ifndef OPTIMIZE_H
#define OPTIMIZE_H

#include <opencv2/opencv.hpp>

namespace eit {

    //------------------------------------------------------------------------------
    // Function: basic_operation
    // Description:
    //   Performs a basic pixel-wise operation on three input matrices.
    //   Each pixel is computed as:
    //      output = abs(input - (input_before - desired))
    //   All input matrices must have the same size and type.
    // Returns:
    //   A new cv::Mat containing the result.
    //------------------------------------------------------------------------------
    cv::Mat basic_operation(const cv::Mat& input,
                            const cv::Mat& input_before,
                            const cv::Mat& desired);

} // namespace eit

#endif //OPTIMIZE_H
