#ifndef FILTERS
#define FILTERS

#include<opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>

#include<boost/thread.hpp>
#include<boost/bind.hpp>
#include<boost/shared_ptr.hpp>
#include<boost/lexical_cast.hpp>
#include<boost/program_options.hpp>
#include<boost/tokenizer.hpp>
#include<boost/math/constants/constants.hpp>
#include<boost/math/special_functions.hpp>
#include<boost/random.hpp>
#include<boost/random/normal_distribution.hpp> 

#include<iostream>
#include<string>
#include<fstream>
#include<chrono>
#include<complex>
#include<functional>

#include "consts.hpp"

using namespace std;
using namespace cv;

typedef std::function<void(cv::Mat&)> holo_filter;
typedef typename cv::MatIterator_<Vec3f> Iterator;

typedef unsigned int row;
typedef unsigned int col;

extern holo_filter filter_remavg;
extern holo_filter filter_sinc;
extern holo_filter filter_rpn;
extern holo_filter filter_rpn_ocv;
extern holo_filter filter_laplacian;
extern holo_filter filter_clear_center;
extern holo_filter filter_real_laplacian;
extern holo_filter filter_inverse_fourier_kernel_one;
extern holo_filter filter_spherical;
extern holo_filter filter_checker;
extern holo_filter filter_eit_lpl;
extern holo_filter filter_linear_phase_load;

#endif
