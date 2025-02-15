#ifndef GERCH_HPP
#define GERCH_HPP

//STD
#include<iostream>
#include<string>
#include<fstream>
#include<chrono>
#include<complex>
#include<mutex>
#include<cstring>
#include<vector>

//OPENCV
#include<opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>

//BOOST
#include<boost/thread.hpp>
#include<boost/bind.hpp>
#include<boost/shared_ptr.hpp>
#include<boost/lexical_cast.hpp>
#include <boost/program_options.hpp>
#include <boost/tokenizer.hpp>

//GSL
// #include <gsl/gsl_math.h>
// #include <gsl/gsl_rng.h>
// #include <gsl/gsl_randist.h>
// #include <gsl/gsl_vector.h>
// #include <gsl/gsl_blas.h>
// #include <gsl/gsl_multifit_nlin.h>

//DLIB
#include <dlib/optimization.h>

//EIT
#include "consts.hpp"
#include "fft.hpp"
#include "filters.hpp"
#include "optimize.hpp"

class gerch{
  //Implementation of the Gerchberg-Saxton algorithm
  eit_hologram _holo;
public:
  gerch();
  ~gerch();

  bool error_less_than(cv::Mat& padded, cv::Mat& initial);
  void operator()(cv::Mat& input,
		  cv::Mat& output);
  cv::Mat eit_lsq(cv::Mat& in,
		  cv::Mat& cmp);
};

#endif
