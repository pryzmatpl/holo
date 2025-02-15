#ifndef REFERENCES
#define REFERENCES

#include<opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<boost/math/constants/constants.hpp>
#include<boost/math/special_functions.hpp>

#include<iostream>
#include<string>
#include<fstream>
#include<chrono>
#include<complex>
#include<functional>

#include "../include/filters.hpp"
#include "../include/consts.hpp"
#include "../include/fft.hpp"

float wavelength=632.8e-9f; //TODO: I shiver upon this global

cv::Mat eit_hologram::holoeye_movavg(cv::Mat& input){
  int rowmid = input.rows/2+1;
  int colmid = input.cols/2+1;
  
  Mat returnee(input.size(),
	       input.type());

  for(int row = 1; row < input.rows-1; row++){
    for(int col = 1; col < input.cols-1; col++){
      //We start from the first feasible thing to do
      float average = 
	[&](int _row, int _col){
	float ret = 0.0f;
	for(int roi_row=-1; roi_row<=1; roi_row++){
	  for(int roi_col=-1; roi_col<=1; roi_col++){
	    float phase_val = input.at<float>(row+roi_row,col+roi_col);
	    ret += phase_val;
	    ret /= 2;
	  }
	}
	return ret;
      }(row,col);

      float pxl_val = input.at<float>(row,col);
      float& final_pxl_value = returnee.at<float>(row,col);
      final_pxl_value = fabs( pxl_val - average ) ; //in principle, remove the average of a 3x3 neighbourhood
    }
  }
  
  return returnee;
}

cv::Mat eit_hologram::holoeye_ref_wavefront_squareaperture(cv::Mat& input,
							   float distance){
  float area = holoeye_delta_x*holoeye_width*holoeye_delta_x*holoeye_height;
  float ref_index = 1.44f;
  float distance_ = (distance == glob_distance) ? distance : glob_distance;
  float wavenum = 2*pi/wavelength;
  float omega = c/wavelength;
  
  int rowmid = input.rows/2+1; 
  int colmid = input.cols/2+1;

  for(int row=0; row<input.rows; row++){
    for(int col=0; col<input.cols; col++){
      Scalar_<float>& val = input.at<Scalar_<float>>(row,col);
      complex<float> _cal(val[0],val[1]);

      float xangle,yangle;
      xangle = atan(holoeye_delta_x*row/distance);
      yangle = atan(holoeye_delta_x*col/distance);
      
      _cal *= area*(exp(neg_I*wavenum*distance)/distance)*(sin(xangle)/xangle)*(sin(yangle)/yangle);
      val[0] = _cal.real();
      val[1] = _cal.imag();
    }
  }

  cout<<"holoeye_ref_wavefront_squareaperture( ";
  
  return input;
}



void eit_hologram::holoeye_ref_wavefront_flat_phase(Mat& input,
						    float object_plane_angle){
  //Reference wavefront for the holoeye modulator for Fresnel mode
  //As written in "Fresnel and Fourier digital holography architectures: a comparison." by
  //Damien P et. al,
  cv::Mat planes_Hw[] = {cv::Mat::zeros(input.size(), CV_32F),
			 cv::Mat::zeros(input.size(), CV_32F)};

  //Filled with real part of the image
  if(input.channels() == 2){
    split(input,planes_Hw);
  }else{
    planes_Hw[0] = input;
  }

  int rowmid = input.rows/2+1;
  int colmid = input.cols/2+1;

  for(int row=0; row<input.rows; row++){
    for(int col=0; col<input.cols; col++){
      complex<float> val = std::polar(1.0f, object_plane_angle);
      
      float& re = planes_Hw[0].at<float>(row,col);
      float& im = planes_Hw[1].at<float>(row,col);
      
      complex<float> inp = complex<float>(re,im);
      inp *= val;

      re = inp.real();
      im = inp.imag();
    }
  }

  if(input.channels() == 2){
    merge(planes_Hw,2,input);
  }else{
    input=planes_Hw[0];
  }
  
  cout<<"holoeye_ref_wavefront_flat_phase ( ";
}


void eit_hologram::holoeye_ref_wavefront_focal(Mat& input){
  //Reference wavefront for the holoeye modulator for Fresnel mode
  //As written in "Fresnel and Fourier digital holography architectures: a comparison." by
  //Damien P et. al,
  cv::Mat planes_Hw[] = {cv::Mat::zeros(input.size(), CV_32F),
			 cv::Mat::zeros(input.size(), CV_32F)};

  normalize(input,input,0,1,CV_MINMAX);
  
  //Filled with real part of the image
  if(input.channels() == 2){
    split(input,planes_Hw);
  }else{
    planes_Hw[0] = input;
  }

  int rowmid = input.rows/2+1;
  int colmid = input.cols/2+1;

  for(int row=0; row<input.rows; row++){
    for(int col=0; col<input.cols; col++){
      float& re = planes_Hw[0].at<float>(row,col);
      float& im = planes_Hw[1].at<float>(row,col);
      
      std::complex<float> val = exp((neg_I*pi*((row-rowmid)*(row-rowmid)*(holoeye_delta_x))
				     *((col-colmid)*(col-colmid)*(holoeye_delta_x)))/
				    (wavelength*glob_distance));

      complex<float> inp = complex<float>(re,im);
      inp *= val;

      re = inp.real();
      im = inp.imag();
    }
  }

  if(input.channels() == 2){
    merge(planes_Hw,2,input);
  }else{
    input=planes_Hw[0];
  }
  
  cout<<"holoeye_ref_wavefront_focal( ";
}

cv::Mat eit_hologram::holoeye_ref_wavefront_sommerfeld(Mat& input,
						       float distance){
  cv::Mat planes_Hw[] = {cv::Mat::zeros(input.size(), CV_32F),
			 cv::Mat::zeros(input.size(), CV_32F)};
      //Filled with real part of the image
  if(input.channels() == 2){
    split(input,planes_Hw);
  }else{
    planes_Hw[0] = input;
  }

  int rowmid = planes_Hw[0].rows/2+1;
  int colmid = planes_Hw[0].cols/2+1;
  
  float RC = glob_distance*sqrt(holoeye_width*holoeye_height*holoeye_delta_x*holoeye_delta_x); 
  //Radius of curvature influences the image A LOT

  for(int row = 0; row < planes_Hw[0].rows; row++){
    for(int col = 0; col < planes_Hw[0].cols; col++){
      float& im = planes_Hw[1].at<float>(row,col);
      float& re = planes_Hw[0].at<float>(row,col);
      
      complex<float> test = (_I*RC)*exp(_I*2.0f*pi/(wavelength*sqrt(RC*RC+(row-rowmid)*(row-rowmid)*holoeye_delta_x*holoeye_delta_x+
								    (col-colmid)*(row-colmid)*holoeye_delta_x*holoeye_delta_x)))/
	((RC*RC)+(row-rowmid)*(row-rowmid)*holoeye_delta_x*holoeye_delta_x+
	 (col-colmid)*(col-colmid)*holoeye_delta_x*holoeye_delta_x);
      
      std::complex<float> vas = std::polar(1.0f, atan(test.imag()/test.real()));
      std::complex<float> imp(re,im);

      imp*=vas;
      
      re = imp.real();
      im = imp.imag();
    }
    //After this loop, we should have a matrix with the Rayleigh-Sommerfeld spherical wave superposed in the phase
    //plane of the image
  }

  cout<<"holoeye_ref_wavefront_sommerfeld( ";
  
  if(input.channels()==2){
    merge(planes_Hw, 2, input);
  }else{
    input = planes_Hw[0];
  }
  return input;
}


Mat eit_hologram::holoeye_ref_wavefront_phase(Mat& input){
  cv::Mat planes_Hw[] = {cv::Mat::zeros(input.size(), CV_32F),
			 cv::Mat::zeros(input.size(), CV_32F)};

  //Filled with real part of the image
  if(input.channels() == 2){
    split(input,planes_Hw);
  }else{
    planes_Hw[0] = input;
  }

  int rowmid = planes_Hw[0].rows/2+1;
  int colmid = planes_Hw[0].cols/2+1;

  for(int row=0; row<input.rows; row++){
    for(int col=0; col<input.cols; col++){

      float& re = planes_Hw[0].at<float>(row,col);
      float& im = planes_Hw[1].at<float>(row,col);
      
      complex<float> phase_ = exp((neg_I*pi/wavelength*(col*col*holoeye_delta_x)));
      complex<float> number_ = complex<float>(re,im);

      number_*=phase_;

      re = number_.real();
      im = number_.imag();
    }
  }

  cout<<"holoeye_ref_wavefront_phase( ";

  if(input.channels()==2){
    merge(planes_Hw, 2, input);
  }else{
    input = planes_Hw[0];
  }
  return input;
}

Mat eit_hologram::holoeye_ref_wavefront_phase2(Mat& input){
  Mat returnee(input);

  for(int row=0; row<input.rows; row++){
    for(int col=0; col<input.cols; col++){

      float& img_val = input.at<float>(row,col);
      complex<float> phase_ = exp((neg_I*pi/wavelength)*sqrt(col*col*holoeye_delta_x*holoeye_delta_x));
      complex<float> number_ = std::polar(img_val,arg(phase_));

      float& ref_ = returnee.at<float>(row,col);
      ref_ = std::arg(number_);
    }
  }
  cout<<"holoeye_ref_wavefront_phase2( ";
  return returnee;
}


cv::Mat eit_hologram::holoeye_chirp(cv::Mat& input, float distance_){
  cv::Mat planes_Hw[] = {cv::Mat::zeros(input.size(), CV_32F),
			 cv::Mat::zeros(input.size(), CV_32F)};

  normalize(input,input,0,1,CV_MINMAX);
  
  //Filled with real part of the image
  if(input.channels() == 2){
    split(input,planes_Hw);
  }else{
    planes_Hw[0] = input;
  }

  int rowmid = input.rows/2+1;
  int colmid = input.cols/2+1;

  float distance = (distance_ == glob_distance) ? distance_ : glob_distance;
  
  for(int row=0; row<input.rows; row++){
    for(int col=0; col<input.cols; col++){
      complex<float> val = complex<float>(exp((neg_I*2.0f*pi/(wavelength*distance))*((row-rowmid)*(row-rowmid)*holoeye_delta_x*holoeye_delta_x+
										     (col-colmid)*(col-colmid)*holoeye_delta_x*holoeye_delta_x)));
      float& re = planes_Hw[0].at<float>(row,col);
      float& im = planes_Hw[1].at<float>(row,col);
      
      complex<float> inp = complex<float>(re,im);

      inp*=val;

      re = inp.real();
      im = inp.imag();
    }
  }

  if(input.channels() == 2){
    merge(planes_Hw,2,input);
  }else{
    input=planes_Hw[0];
  }
  
  cout<<"holoeye_chirp( ";
  return input;
}


cv::Mat eit_hologram::holoeye_reference(cv::Mat& input,
					std::string wf_type){
  //This is only supposed to merge the reference phase to the real values 
  //of the images, and then return the actual reference wave
  try{
    normalize(input,input,0,1,CV_MINMAX);
    
    if(wf_type == "FLAT"){
      holoeye_ref_wavefront_flat_phase(input);
      cout<<"holoeye_ref_wavefront_flat_phase (";
      qshow(input,"FLAT");
    }else if(wf_type == "REF"){
      holoeye_ref_wavefront_phase(input);
      cout<<"holoeye_ref_wavefront_phase (";
      qshow(input,"REF");
    }else if(wf_type == "REFFOCAL"){
      holoeye_ref_wavefront_focal(input);
      cout<<"holoeye_ref_wavefront_focal (";
      qshow(input,"REFFOCAL");
    }else if(wf_type == "SQUARE"){
      holoeye_ref_wavefront_squareaperture(input);
      cout<<"holoeye_ref_wavefront_squareaperture (";
      qshow(input,"SQUARE");
    }else if(wf_type == "REF2"){
      holoeye_ref_wavefront_phase2(input);
      cout<<"holoeye_ref_wavefront_phase2 (";
      qshow(input,"REF2");
    }else if(wf_type == "RAYLEIGH"){
      holoeye_ref_wavefront_sommerfeld(input);
      cout<<"holoeye_ref_wavefront_sommerfeld (";
      qshow(input,"RAYLEIGH");
    }else if(wf_type == "CHIRP"){
      holoeye_chirp(input);
      cout<<"holoeye_chirp (";
      qshow(input,"CHIRP");
    }else{
      
    };
    
    normalize(input,input,0,1,CV_MINMAX);
    cout<<flush;
    return input; //return the phase information generated for the image
  }catch(std::exception &e){
    cout<<"Exception in holoeye_reference(Mat,Mat,string) -"<<e.what()<<"\n";
  }
}



#endif
