#include "optimize.hpp"
#include <stdexcept>
#include <cmath>

namespace eit {

  cv::Mat basic_operation(const cv::Mat& input,
                          const cv::Mat& input_before,
                          const cv::Mat& desired) {
    // Ensure all matrices have the same size and type.
    if (input.size() != input_before.size() || input.size() != desired.size()) {
      throw std::invalid_argument("All input matrices must have the same size");
    }
    if (input.type() != input_before.type() || input.type() != desired.type()) {
      throw std::invalid_argument("All input matrices must have the same type");
    }

    // Create an output matrix (clone the header from input).
    cv::Mat output = input.clone();

    int channels = input.channels();
    int rows = input.rows;
    // Total number of elements per row (all channels).
    int totalCols = input.cols * channels;

    // Iterate over each row.
    for (int row = 0; row < rows; row++) {
      // Pointers to the raw data for the current row.
      const uchar* ptr_input    = input.ptr<uchar>(row);
      const uchar* ptr_before   = input_before.ptr<uchar>(row);
      const uchar* ptr_desired  = desired.ptr<uchar>(row);
      uchar*       ptr_output   = output.ptr<uchar>(row);

      // Process each element.
      for (int col = 0; col < totalCols; col++) {
        // Only process for col > 1 (as in original code).
        if (col > 1) {
          int diff = static_cast<int>(ptr_input[col]) -
                     (static_cast<int>(ptr_before[col]) - static_cast<int>(ptr_desired[col]));
          ptr_output[col] = static_cast<uchar>(std::abs(diff));
        }
      }
    }

    return output;
  }

} // namespace eit
