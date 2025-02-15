#include "../include/gerch.hpp"

namespace po = boost::program_options;
using namespace dlib;
using namespace std;

gerch::gerch(){
}

gerch::~gerch(){
}

bool gerch::error_less_than(cv::Mat& in,
			    cv::Mat& initial){
}


void amphasetoreim(cv::Mat input[2]){
  for(int row=0; row<input[0].rows; row++){
    for(int col=0; col<input[0].cols; col++){
      float& re = input[0].at<float>(row,col); //which right now is sqrt(intensity)
      float& im = input[1].at<float>(row,col); //which right now is random phase

      float real = re*cos(im); //the real part is the amplitude times the cos of phase
      float imaginary = im*sin(im);

      re = real;
      im = imaginary;
    }
    //Right now we have the intensity and the phase encoded in complex numbers
  }
}

void reimtoamphase(cv::Mat input[2]){
  cv::Mat ampl,phases;
  phase(input[0], input[1], phases);
  magnitude(input[0], input[1], ampl);
  input[0] = ampl;
  input[1] = phases;
  //Front to back between re-im representation and ampl-phase
}

void gerch::operator()(cv::Mat& input,
		       cv::Mat& output){

  std::cout<<"Going into GS alg\n";

  cv::Mat input_cplx[] = {cv::Mat::zeros(input.size(), CV_32F),
			  cv::Mat::zeros(input.size(), CV_32F)}; //empty for phase
  typetest(input);
  typetest(input_cplx[0]);
  
  cv::Mat intensity_sqrt = cv::Mat::zeros(input.size(), CV_32F);
  cv::Mat singles        = cv::Mat::zeros(input.size(), CV_32F);
  cv::Mat phase_temp     = cv::Mat::zeros(input.size(), CV_32F);

  for(int row=0; row<input.rows; row++){ //opencv's sqrt does not do what I want
    for(int col=0; col<input.cols; col++){
      float& ref = intensity_sqrt.at<float>(row,col);
      float& sing= singles.at<float>(row,col);
      sing = 1.0f;
      ref = 1.0f;
    }
  }//fill the ampl or sqrt(intens) 

  cv::Mat first_iter(input.size(), CV_32FC1);
  typetest(first_iter);
  typetest(input_cplx[0]);
  typetest(input_cplx[1]);
  
  cv::randn(input_cplx[1],128,32); //random phase distribution

  //According to Fienup, we have to use the amplitude
  //and not the phase as our estimation variable

  //TODO:
  // - analyze the code from the student's GS implementation
  // - look into classical machine learning (steepest descent, lsq etc) to see a possibility of calculating the holograms
  // - test what happens when adding a Least Squares test between the input data and the approximation
  // - test if the domains are correctly analyzed and the proper mode is input into the iterations
  // - backpropagation? machine learning? deep learning?
  
  for(int a = 0; a < 100000; a++){
    amphasetoreim(input_cplx);                            //we have re-im representation
    merge(input_cplx, 2, first_iter);                     //merge the two channels, [0] = uniform amplitude, [1] = random phase
    dft(first_iter,first_iter);                           //first DFT from Fourier space to object space
    split(first_iter, input_cplx);                        //split the DFT channels into the input_cplx

    reimtoamphase(input_cplx);                            //return from re-im to am-phase in the spatial frequency domain
    intensity_sqrt.copyTo(input_cplx[0]);                 //copy uniform array of 1s in Fourier space into the amplitudes, leave phase alone

    amphasetoreim(input_cplx);                            //go from amphase to reim
    merge(input_cplx, 2, first_iter);                     //merge into dft-capable device
    dft(first_iter,first_iter,DFT_INVERSE);               //go back to linear space
    split(first_iter,input_cplx);                         //split channels again

    reimtoamphase(input_cplx);                            //reim to amphase in order to copy the 1s into Fourier space again
    singles.copyTo(input_cplx[0]);                        //copy uniform amplitude into fourier space
  }

  merge(input_cplx, 2, output);
	
}

cv::Mat gerch::eit_lsq(cv::Mat& in,
		       cv::Mat& cmp){
  cv::Mat returnee;
  
  if( !cv::solve(in,cmp,returnee,DECOMP_NORMAL) ){
    throw "Not possible to solve the problem";
  }
  
  return returnee;
}
