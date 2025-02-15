#ifndef GERCH_HPP
#define GERCH_HPP

// STL
#include <iostream>
#include <string>
#include <fstream>
#include <chrono>
#include <complex>
#include <mutex>
#include <cstring>
#include <vector>

// OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

// Boost
#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>
#include <boost/tokenizer.hpp>

// DLIB
#include <dlib/optimization.h>

// EIT project headers
#include "consts.hpp"
#include "fft.hpp"
#include "filters.hpp"
#include "optimize.hpp"

namespace eit {

    //------------------------------------------------------------------------------
    // Class: gerch
    // Description: Implements the Gerchberg-Saxton algorithm for hologram synthesis.
    //------------------------------------------------------------------------------
    class gerch {
        // Private member: instance of the hologram processing class.
        eit_hologram _holo;

    public:
        // Constructor and destructor.
        gerch();
        ~gerch();

        // Returns true if the error between 'padded' and 'initial' is below a threshold.
        bool error_less_than(cv::Mat& padded, cv::Mat& initial);

        // Operator overload to process 'input' and generate an output hologram.
        void operator()(cv::Mat& input, cv::Mat& output);

        // Performs a least-squares optimization between input 'in' and comparison 'cmp'.
        cv::Mat eit_lsq(cv::Mat& in, cv::Mat& cmp);
    };

} // namespace eit

#endif // GERCH_HPP
