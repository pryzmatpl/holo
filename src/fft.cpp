#include "../include/fft.hpp"

/* 
 * based on "Fast algorithms for free-space diffraction patterns calculation"
 * by David Mas et. al.
 * We CANNOT use Fraunhoffer diffraction, due to the fact that the conditions
 * are not met ! Our image and source would have to be 81 m apart from the 
 * aperture. But why not test it? :)
 * We treat our modulator as an aperture, with the laser acting as a source
 * that will be well aligned, just so that we will be able to calculate the
 * hologram.
 */

using namespace cv;
using namespace std;
#include <complex>
#include <iostream>
#include <valarray>
#include <algorithm>

#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp> 
#include <boost/thread.hpp>

using namespace cv;
extern float wavelength; //TODO: I shiver upon this global

void typetest(cv::Mat& in){
    cout<<in.size()<<","<<in.type()<<","<<in.channels()<<endl;
}

template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

float frft_angle;

Mat& operator*(Mat& lhs, std::complex<float> rhs){
  for(int row = 0; row < lhs.rows; row++){
    for(int col = 0; col < lhs.cols; col++){
      Vec3b& x = lhs.at<Vec3b>(row,col);
      
      if(lhs.channels()==1){
	x[0]*=rhs.real();
      }else if(lhs.channels()==2){
	x[0]*=rhs.real();
	x[1]*=rhs.imag();
      }else{
	throw "Channels do not match in operator*(Mat&,complex<float>)";
      }
    }
  }
  return lhs;
}

Mat& operator*=(Mat& lhs, std::complex<float> rhs){
  for(int row = 0; row < lhs.rows; row++){
    for(int col = 0; col < lhs.cols; col++){
      Vec3b& x = lhs.at<Vec3b>(row,col);
      
      if(lhs.channels()==1){
	x[0]*=rhs.real();
      }else if(lhs.channels()==2){
	x[0]*=rhs.real();
	x[1]*=rhs.imag();
      }else throw "Channels do not match in operator*(Mat&,complex<float>)";
    }
  }
  return lhs;
}

bool operator<(Size lhs, Size rhs){
  return (lhs.width*lhs.height) < (rhs.width*rhs.height) ;
}

Mat operator*(Mat& lhs, Mat& rhs){
  //TODO: fix this, what if we have different # channels or width/height
  Mat returnee = Mat::zeros(lhs.size(), lhs.type());

  for(int row = 0; row < lhs.rows; row++){
    for(int col = 0; col < lhs.cols; col++){
      Vec3b& retval = returnee.at<Vec3b>(row,col);
      Vec3b& lhsval = lhs.at<Vec3b>(row,col);
      Vec3b& rhsval = rhs.at<Vec3b>(row,col);
      
      for(int ch=0; ch<retval.channels; ch++){
	retval[ch] = lhsval[ch]*rhsval[ch];
      }
    }
  }
  return returnee;
}

//Tiling function
cv::Mat eit_hologram::tile_to_fhd(Mat& input){
  try{
    Mat magI;

    if(input.channels() != 1){
      cv::Mat planes_Hw[] = {cv::Mat::zeros(input.size(), CV_32F),
			     cv::Mat::zeros(input.size(), CV_32F)}; //Filled with real part of the image
      split(input,planes_Hw);
      
      phase(planes_Hw[0], planes_Hw[1], magI);
    }else{
      magI = input;
    }

    magI += Scalar::all(1);                    // switch to logarithmic scale
    // crop the spectrum, if it has an odd number of rows or columns
    magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));
    
    // rearrange the quadrants of Fourier image  so that the origin is at the image center
    int cx = magI.cols/2;
    int cy = magI.rows/2;
    
    //TODO : SEE WHAT UNSWAPPING QUADRANTS DOES
    cv::Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    cv::Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
    cv::Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
    cv::Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right
    cv::Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)

    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);
     
    normalize(magI, magI, 0, 1, CV_MINMAX); // Transform the matrix with float values into a
    // viewable image form (float between values 0 and 1).

    Mat tiled( Size((int)holoeye_width,(int)holoeye_height),
	       magI.type());

    for(int row = 0; row < tiled.rows; row++){
      for(int col=0; col<tiled.cols; col++){
	if((row%input.rows==0) && (col%input.cols==0)){
	  if((row+input.rows>holoeye_height)&&((col+input.cols)>holoeye_width)){
	    Mat roi(tiled,Rect(col, row, holoeye_width-col, holoeye_height-row));
	    Mat roi_magI(magI, Rect(0,0,holoeye_width-col, holoeye_height-row));
	    roi_magI.copyTo(roi);
	  }else if((col+magI.cols>holoeye_width)){
	    Mat roi(tiled,Rect(col, row, holoeye_width-col, magI.rows));
	    Mat roi_magI(magI, Rect(0,0,holoeye_width-col, magI.rows));
	    roi_magI.copyTo(roi);
	  }else if((row+magI.rows)>holoeye_height){
	    Mat roi(tiled,Rect(col, row, magI.cols, holoeye_height-row));
	    Mat roi_magI(magI, Rect(0,0,magI.cols,holoeye_height-row));
	    roi_magI.copyTo(roi);
	  }else{
	    Mat roi(tiled,Rect(col, row, magI.cols, magI.rows));
	    magI.copyTo(roi);
	  }
	}
      }
    }
    normalize(tiled, tiled, 0, 1, CV_MINMAX); // Transform the matrix with float values into a
    return tiled;
  }catch(std::exception& e){
    cout<<"Error in tile_to_fhd -"<<e.what()<<"\n";
  }
}

cv::Mat eit_hologram::tile_to_fhd_amp(Mat& input){
  try{
    cv::Mat planes_Hw[] = {cv::Mat::zeros(input.size(), CV_32F),
			   cv::Mat::zeros(input.size(), CV_32F)}; //Filled with real part of the image
    split(input,planes_Hw);
    
    Mat magI;
    magnitude(planes_Hw[0], planes_Hw[1], magI);

    magI += Scalar::all(1);                    // switch to logarithmic scale
    // crop the spectrum, if it has an odd number of rows or columns
    magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));
    
    // rearrange the quadrants of Fourier image  so that the origin is at the image center
    int cx = magI.cols/2;
    int cy = magI.rows/2;

    //TODO : SEE WHAT UNSWAPPING QUADRANTS DOES
    
    cv::Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    cv::Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
    cv::Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
    cv::Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right
    cv::Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);
    
    normalize(magI, magI, 0, 1, CV_MINMAX); // Transform the matrix with float values into a
    // viewable image form (float between values 0 and 1).

    Mat tiled( Size((int)holoeye_width,(int)holoeye_height),
	       magI.type());

    for(int row = 0; row < tiled.rows; row++){
      for(int col=0; col<tiled.cols; col++){
	if((row%input.rows==0) && (col%input.cols==0)){
	  if((row+input.rows>holoeye_height)&&((col+input.cols)>holoeye_width)){
	    Mat roi(tiled,Rect(col, row, holoeye_width-col, holoeye_height-row));
	    Mat roi_magI(magI, Rect(0,0,holoeye_width-col, holoeye_height-row));
	    roi_magI.copyTo(roi);
	  }else if((col+magI.cols>holoeye_width)){
	    Mat roi(tiled,Rect(col, row, holoeye_width-col, magI.rows));
	    Mat roi_magI(magI, Rect(0,0,holoeye_width-col, magI.rows));
	    roi_magI.copyTo(roi);
	  }else if((row+magI.rows)>holoeye_height){
	    Mat roi(tiled,Rect(col, row, magI.cols, holoeye_height-row));
	    Mat roi_magI(magI, Rect(0,0,magI.cols,holoeye_height-row));
	    roi_magI.copyTo(roi);
	  }else{
	    Mat roi(tiled,Rect(col, row, magI.cols, magI.rows));
	    magI.copyTo(roi);
	  }
	}
      }
    }
    normalize(tiled, tiled, 0, 1, CV_MINMAX); // Transform the matrix with float values into a
    return tiled;
  }catch(std::exception& e){
    cout<<"Error in tile_to_fhd -"<<e.what()<<"\n";
  }
}


cv::Size eit_hologram::set_optimal_holosize(Mat& input, Mat& output){
  //Take a real image and then convert it into a two channel matrix
  try{
    normalize(input,input,0,1,CV_MINMAX);

    Mat padded;
    int m = getOptimalDFTSize( 2*input.rows );
    int n = getOptimalDFTSize( 2*input.cols );    // on the border add zero values

    flip(input,input,1);
    
    copyMakeBorder(input,
		   padded,
		   0, m/2, n/2, 0,
		   BORDER_CONSTANT, Scalar::all(0));

    cv::Mat planes_Hw[] = {cv::Mat_<float>(padded), 
			   cv::Mat::zeros(padded.size(), CV_32F)}; //Filled with real part of the image
  
    Mat returnee;
    merge(planes_Hw, 2, returnee);

    returnee.convertTo(output, returnee.type());
    normalize(input,input,0,1,CV_MINMAX);
    return Size(m,n);
  }catch(std::exception &e){
    cout<<"Exception in set_optimal_holosize(Mat&, Mat&) - "<<e.what()<<"\n";
  }
}

//The genereal transform function
cv::Mat eit_hologram::holoeye_transform(cv::Mat& padded,
					cv::Mat& adft_data,
					std::string holotype){
  assert(padded.channels() == 2);

  try{
    if(holotype == "FRESNEL"){
      return holoeye_fresnel(padded,
			     adft_data);
    }else if(holotype == "FRFT"){
      return holoeye_fractional_ft(padded,
				   adft_data,
				   frft_angle);
    }else if(holotype ==  "RPN"){
      return holoeye_rpn(padded,
			 adft_data);
    }else if(holotype == "FFT"){
      return holoeye_dft(padded,
			 adft_data);
    }else if(holotype == "FRECONV"){
      return holoeye_convolution_extended(padded,
					  adft_data);
    }else if(holotype == "FRFT"){
      return holoeye_frft(padded,
			  adft_data,
			  0.3f);
    }else if(holotype == "AS"){
      return holoeye_angular_spectrum(padded,
				      adft_data);
    }else{
      return holoeye_fresnel(padded,
			     adft_data);
    }
  }catch(std::exception &e){
    cout<<"Exception in transform(Mat,Mat,string) -"<<e.what()<<"\n";
  }
}

cv::Mat eit_hologram::holoeye_fractional_ft(cv::Mat& input, 
			      cv::Mat& cplx,
			      float angle){
  //from almeida, chirp->ft->chirp->cplx-amp
  try{  
    Mat_<float> inputdat [] = {cv::Mat::zeros(input.size(), input.type()),
			       cv::Mat::zeros(input.size(), input.type())}; //just give the phase from the chirp function
    
    int rowmid = input.rows/2+1;
    int colmid = input.cols/2+1;
    normalize(input,input,0,1,CV_MINMAX);
    split(input,inputdat);

    //chirp 1
    for(int row=0; row<input.rows; row++){
      for(int col=0; col<input.cols; col++){
    	float& in_re = inputdat[0].at<float>(row,col);
    	float& in_im = inputdat[1].at<float>(row,col);

    	complex<float> indat(in_re,in_im);

	indat *= exp(_I*((row*row*holoeye_delta_x*holoeye_delta_x+col*col*holoeye_delta_x*holoeye_delta_x)/2.0f)*tan(pi/2.0f-angle));

	//scaling of fft input
	float f_x = 1.0f/(([&](){return ((row)==0) ? 1.0f : row;}())*holoeye_delta_x);
	float f_y = 1.0f/(([&](){return ((row)==0) ? 1.0f : row;}())*holoeye_delta_x);

	indat *= exp(_I*(row*holoeye_delta_x+col*holoeye_delta_x)*(f_x+f_y)*(1.0f/sin(angle)));

	in_re = indat.real();
	in_im = indat.imag();
      }
    }

    merge(inputdat, 2, input);
    dft(input,input);
    split(input,inputdat);

    for(int row=0; row<input.rows; row++){
      for(int col=0; col<input.cols; col++){
    	float f_x = 1.0f/(([&](){return ((row)==0) ? 1.0f : row;}())*holoeye_delta_x);
    	float f_y = 1.0f/(([&](){return ((row)==0) ? 1.0f : row;}())*holoeye_delta_x);
	
    	complex<float> h_a = exp(_I*((f_x+f_y)/2.0f)*tan(pi/2.0f-angle))*sqrt((1.0f-_I*tan(pi/2.0f-angle))/(2.0f*pi));
	
    	float& in_re = inputdat[0].at<float>(row,col);
    	float& in_im = inputdat[1].at<float>(row,col);

    	complex<float> indat(in_re,in_im);
    	indat *= h_a;
	
    	in_re = indat.real();
	in_im = indat.imag();
      }
    }

    merge(inputdat,2,input);
    merge(inputdat,2,cplx);

    return input;
  }catch(std::exception& e){
    cout<<"Fractional FT(): "<< e.what() <<"\n";
  }
}


cv::Mat eit_hologram::holoeye_angular_spectrum(Mat& input,
					       Mat& cplx){
  //CWO based angular spectrum method
  //assert(input.channels == 2);
  try{  
    Mat_<float> inputdat [] = {cv::Mat::zeros(input.size(), input.type()),
			       cv::Mat::zeros(input.size(), input.type())}; //just give the phase from the chirp function

    int rowmid = input.rows/2+1;
    int colmid = input.cols/2+1;

    normalize(input,input,0,1,CV_MINMAX);
    dft(input,input);
    split(input,inputdat);
    
    for(int row=0; row<input.rows; row++){
      for(int col=0; col<input.cols; col++){
    	float f_x = 1.0f/(([&](){return ((row-rowmid)==0) ? 1.0f : row-rowmid;}())*holoeye_delta_x);
    	float f_y = 1.0f/(([&](){return ((row-rowmid)==0) ? 1.0f : row-rowmid;}())*holoeye_delta_x);
	
    	complex<float> h_a = exp(_I*glob_distance*sqrt((2.0f*pi/wavelength)*(2.0f*pi/wavelength)-
    						       4.0f*pi*pi*(f_x*f_x+f_y*f_y)));
	
    	h_a *= exp(neg_I*2.0f*pi*((row-rowmid)*f_x+(col-colmid)*f_y));

    	float& in_re = inputdat[0].at<float>(row,col);
    	float& in_im = inputdat[1].at<float>(row,col);

    	complex<float> indat(in_re,in_im);
    	indat *= h_a;
	
    	in_re = indat.real();
    	 in_im = indat.imag();
      }
    }

    merge(inputdat,2,input);
    merge(inputdat,2,cplx);

    return input;
  }catch(std::exception& e){
    cout<<"Shifted angular spectrum exception(): "<< e.what() <<"\n";
  }
}

//Transform functions:
cv::Mat eit_hologram::holoeye_fresnel(Mat& input, //input image to calculate the hologram
				      Mat& c_data,
				      float distance,
				      float angle){
  try{
    float pitchrow,pitchcol = holoeye_delta_x;

    Mat_<float> h_times_r [] = {cv::Mat::zeros(input.size(), CV_32F),
				cv::Mat::zeros(input.size(), CV_32F)}; //just give the phase from the chirp function
    split(input,h_times_r);
    cv::Mat chirp = holoeye_chirp(h_times_r[0],
				  [&](){ return (distance != glob_distance) ? distance : glob_distance; }() ); 
    
    h_times_r[1] += chirp; //Adding the chirp only to the phase image
  
    Mat_<float> transformed [] = {Mat::zeros(input.size(), input.type()),
				  Mat::zeros(input.size(), input.type())}; //first factor

    Mat complexHR;
    merge(h_times_r, 2, complexHR);
    dft(complexHR,complexHR);

    //Mat HR_times_chirp = complexHR*chirp; //pointwise multiplication of the FFT(chirp) and FFT(image) w/ a flat plane wave
    //split(HR_times_chirp, transformed);
    input = complexHR;
    c_data = complexHR;

    return complexHR;
  }catch(std::exception &e){
    cout<<"Exception in holoeye_fresnel() -" << e.what() <<"\n";
  }
}

cv::Mat eit_hologram::holoeye_convolution_extended(cv::Mat& input, 
						   cv::Mat& cplx,
						   float distance){
  try{
    //Since this is a transform WITH an auxliarry added chirp,
    //We have to be careful what we use in the arguments
    //expand input image to optimal size
    //This is an implementation of the "New Convolution Algorithm"
    //for reconstructing extended objects in digitally rec. holograms
    //Method 1 - modified angular spectrum transfer
    cv::Mat planes_Hw[] = {cv::Mat::zeros(input.size(), CV_32F),
			   cv::Mat::zeros(input.size(), CV_32F)}; //Filled with real part of the image

    //Then generate another convolutionable function to combine both
    float distance_ = [&](){return (distance == glob_distance) ? distance : glob_distance; }();

    Mat ref = holoeye_ref_wavefront_sommerfeld(input,
					       distance);
    input*=ref;

    dft(input,input);

    //Should be the spatial frequencies of the reference wave here, but I'll test a smaller version to see anything...
    //I expect it to not be computationally working
    merge(planes_Hw,2,input);
    merge(planes_Hw,2,cplx);

    return input;
  }catch(std::exception &e){
    cout<<"Exception in fresnel-conv() -"<<e.what()<<"\n";
  }
}

cv::Mat eit_hologram::holoeye_frft(cv::Mat& input, 
				   cv::Mat& cplx,
				   float order){
  try{
    //FRFT by Zaktas
    //V.Ashok Narayanan et. al. , 2003
    
    
    
    return input;
  }catch(std::exception &e){
    cout<<"Exception in fresnel-conv() -"<<e.what()<<"\n";
  }
}


cv::Mat eit_hologram::holoeye_dft(cv::Mat& g_ROI, Mat& outputdata){
  try{
    dft(g_ROI, g_ROI);  // this way the result may fit in the source matrix
    outputdata = g_ROI; //Save the outputdata
    return g_ROI;
  }catch(std::exception& e){
    cout<<"Holoye dft exception-"<<e.what()<<"\n";
  }
}

cv::Mat eit_hologram::holoeye_rpn(cv::Mat& g_ROI, Mat& outputdata){
  //Returns a viewable IDFT holographic image with random gaussian phase
  try{	
    cv::Mat planes[] = {cv::Mat::zeros(g_ROI.size(), CV_32FC1),
			cv::Mat::zeros(g_ROI.size(), CV_32FC1)};
    split(g_ROI, planes);

    cv::Mat planes_idft[] = {cv::Mat_<float>(planes[0]),
			     cv::Mat::zeros(planes[0].size(), CV_32FC1)};

    cv::Mat complexI(g_ROI);

    cv::Scalar mean = cv::mean(complexI);
    float fmean = sum(mean)[0];

    cv::Mat planes_factor[]  = {cv::Mat::zeros(g_ROI.size(),CV_32FC1), 
				cv::Mat::zeros(g_ROI.size(), CV_32FC1)};

    for(int row = 0; row < complexI.rows; row++){
      for(int col = 0; col < complexI.cols; col++){
	float& pxl_re = planes_factor[0].at<float>(row,col);
	float& pxl_im = planes_factor[1].at<float>(row,col);
    
	pxl_re=std::complex<float>(std::complex<float>((float)1/(float)sqrt(1920*1080))*complexI.at<std::complex<float>>(row,col)-std::complex<float>(fmean)).real();
	//According to "Random Phase Textures"
	pxl_im=std::complex<float>(std::complex<float>((float)1/(float)sqrt(1920*1080))*complexI.at<std::complex<float>>(row,col)-std::complex<float>(fmean)).imag();
	//According to "Random Phase Textures"
      }
    }
 
    cv::Mat complex_factor(g_ROI.size(), CV_32FC1);
    merge(planes_factor, 2, complex_factor);         // Add to the expanded another plane with zeros
 
    boost::mt19937 rng; 
    boost::normal_distribution<> nd(0.0,1.0);
    boost::variate_generator<boost::mt19937&, 
			     boost::normal_distribution<> > var_nor(rng, nd);

    cv::Mat planes_gaussian[] = {cv::Mat::zeros(g_ROI.size(), CV_32FC1), 
				 cv::Mat::zeros(g_ROI.size(), CV_32FC1)};

    // //Here we will add the random normal distribution of phase
    for(int row = 0; row < complexI.rows; row++){
      for(int col = 0; col < complexI.cols; col++){
	std::complex<float>& pxl_val = complex_factor.at<std::complex<float>>(row,col); //prep the pixel var
	float& pxl_re = planes_gaussian[0].at<float>(row,col);
	float& pxl_im = planes_gaussian[1].at<float>(row,col);
	float rnd_phase_rad = var_nor(); //first generation of phase
	std::complex<float> val(pxl_re,pxl_im);
	std::complex<float> phasevalue(exp(neg_I*rnd_phase_rad));

	val*=phasevalue;
     
	pxl_re = phasevalue.real();
	pxl_im = phasevalue.imag();
      }
    }

    Mat complex_gaussian(g_ROI.size(), CV_32F);
    merge(planes_gaussian, 2, complex_gaussian);         // Add to the expanded another plane with zeros

    dft(complexI, complexI);            // this way the result may fit in the source matrix
    outputdata = complexI; //Save the outputdata

    split(complex_gaussian, planes_gaussian);
    split(complexI, planes);

    for(int row = 0; row < complexI.rows; row++){
      for(int col = 0; col < complexI.cols; col++){
	float& pxl_re = planes_factor[0].at<float>(row,col);
	float& pxl_im = planes_factor[1].at<float>(row,col);

	float& pxl_re_gauss = planes_gaussian[0].at<float>(row,col);
	float& pxl_im_gauss = planes_gaussian[1].at<float>(row,col);

	complex<float> val(pxl_re, pxl_im);
	complex<float> val_gauss(pxl_re_gauss, pxl_im_gauss);
     
	val_gauss*=val;

	pxl_re_gauss = val.real();
	pxl_im_gauss = val.imag();
      }
    }

    Mat interphase;
    phase(planes_gaussian[0],planes_gaussian[1],interphase);
    Mat chirp = holoeye_chirp(interphase);

    for(int row = 0; row < complexI.rows; row++){
      for(int col = 0; col < complexI.cols; col++){
	float& pxl_re = planes[0].at<float>(row,col);
	float& pxl_im = planes[1].at<float>(row,col);

	float& pxl_re_gauss = planes_gaussian[0].at<float>(row,col);
	float& pxl_im_gauss = planes_gaussian[1].at<float>(row,col);

	complex<float> val(pxl_re, pxl_im);
	complex<float> val_gauss(pxl_re_gauss, pxl_im_gauss);

	float phase_ = chirp.at<float>(row,col);
	complex<float> sinc_filt = std::polar<float>(1.0f, phase_);

	val+=val_gauss;
	val+=sinc_filt;

	pxl_re = val.real();
	pxl_im = val.imag();
      }
    }

    merge(planes,2,outputdata);
    merge(planes,2,g_ROI);

    return outputdata;
    // //According to 3.3.1 of Chap1.Computer Generated Holograms ^
    // //by W.J.Dallas all we have to do is :
    // // 1) Load the object (image in our case)
    // // 2) Apply random phase diffuser
    // // 3) Inverse fourier transform

    // // compute the magnitude and switch to logarithmic scale
    // // => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
  }catch(std::exception &e){
    cout<<"Exception in rpn() -"<<e.what()<<"\n";
  }
}

cv::Mat eit_hologram::holoeye_rpn_no_twin(cv::Mat g_ROI, Mat& outputdata){
  //Returns a viewable DFT holographic image with random gaussian phase
  cv::Mat padded;                            //expand input image to optimal size
  int m = getOptimalDFTSize( g_ROI.rows );
  int n = getOptimalDFTSize( g_ROI.cols ); // on the border add zero values
  copyMakeBorder(g_ROI, padded, 0, m - g_ROI.rows, 0, n - g_ROI.cols, BORDER_CONSTANT, Scalar::all(0));
	
  cv::Mat planes[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32FC1)};
  cv::Mat planes_idft[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32FC1)};
  cv::Mat complexI(padded.size(), CV_32FC1);
  merge(planes, 2, complexI);         // Add to the expanded another plane with zeros

  cv::Scalar mean = cv::mean(complexI);
  float fmean = sum(mean)[0]; //For DC Term removal (not working in the best sense possible) - we go through the intensities and remove the average from all of them

  cv::Mat planes_factor[]  = {cv::Mat::zeros(padded.size(),CV_32FC1), 
			      cv::Mat::zeros(padded.size(), CV_32FC1)}; //Planes factor - phase factor but don't know yet what for

  for(int row = 0; row < complexI.rows; row++){
    for(int col = 0; col < complexI.cols; col++){
      float& pxl_re = planes_factor[0].at<float>(row,col);
      float& pxl_im = planes_factor[1].at<float>(row,col);
    
      pxl_re=std::complex<float>(std::complex<float>((float)1/(float)sqrt(1920*1080))*complexI.at<std::complex<float>>(row,col)-std::complex<float>(fmean)).real(); //According to "Random Phase Textures"
      pxl_im=std::complex<float>(std::complex<float>((float)1/(float)sqrt(1920*1080))*complexI.at<std::complex<float>>(row,col)-std::complex<float>(fmean)).imag(); //According to "Random Phase Textures"
    }
  }
 
  cv::Mat complex_factor(padded.size(), CV_32FC1); //Holds the image pixels with the average removed
  merge(planes_factor, 2, complex_factor);         // Add to the expanded another plane with zeros
 
  boost::mt19937 rng; 
  boost::normal_distribution<> nd(0.0,1.0);
  boost::variate_generator<boost::mt19937&, 
			   boost::normal_distribution<> > var_nor(rng, nd);

  cv::Mat planes_gaussian[] = {cv::Mat::zeros(padded.size(), CV_32FC1), 
			       cv::Mat::zeros(padded.size(), CV_32FC1)};//Hold the random phase distribution over the image

  //Here we will prep the normal distribution of phase
  for(int row = 0; row < complexI.rows; row++){
    for(int col = 0; col < complexI.cols; col++){
      float& pxl_re = planes_gaussian[0].at<float>(row,col);
      float& pxl_im = planes_gaussian[1].at<float>(row,col);
      
      float rnd_phase_rad = var_nor(); //first generation of phase

      std::complex<float> phasevalue(exp(neg_I*rnd_phase_rad));

      pxl_re = phasevalue.real();
      pxl_im = phasevalue.imag();
    }
  }

  Mat complex_gaussian(padded.size(), CV_32F);
  merge(planes_gaussian, 2, complex_gaussian);         // Add to the expanded another plane with zeros

  dft(complexI, complexI); // this way the result may fit in the source matrix
  outputdata = complexI;   //Save the complex outputdata by reference for further use

  split(complex_gaussian, planes_gaussian);
  split(complexI, planes);

  for(int row = 0; row < complexI.rows; row++){
    for(int col = 0; col < complexI.cols; col++){
      float& pxl_re = planes_factor[0].at<float>(row,col);
      float& pxl_im = planes_factor[1].at<float>(row,col);

      float& pxl_re_gauss = planes_gaussian[0].at<float>(row,col);
      float& pxl_im_gauss = planes_gaussian[1].at<float>(row,col);

      complex<float> val(pxl_re, pxl_im);
      complex<float> val_gauss(pxl_re_gauss, pxl_im_gauss);
     
      val_gauss*=val;

      pxl_re_gauss = val.real();
      pxl_im_gauss = val.imag();
    }
  }

  Mat interphase;
  phase(planes_gaussian[0],planes_gaussian[1],interphase);
  //Adding a sinc filter to remove the DC term
  Mat sinc_filtered = holoeye_chirp(interphase);

  //Adding the image complex wavefront to the random gaussian complex wavefront
  //Adding then the sinc phase component
  for(int row = 0; row < complexI.rows; row++){
    for(int col = 0; col < complexI.cols; col++){
      float& pxl_re = planes[0].at<float>(row,col);
      float& pxl_im = planes[1].at<float>(row,col);

      float& pxl_re_gauss = planes_gaussian[0].at<float>(row,col);
      float& pxl_im_gauss = planes_gaussian[1].at<float>(row,col);

      float phase_ = sinc_filtered.at<float>(row,col);
      complex<float> sinc_filt = std::polar<float>(1.0f, phase_);
      complex<float> val(pxl_re, pxl_im);
      complex<float> val_gauss(pxl_re_gauss, pxl_im_gauss);
     
      val += val_gauss;
      val -= sinc_filt;

      pxl_re = val.real();
      pxl_im = val.imag();
    }
  }
 
  // //According to 3.3.1 of Chap1.Computer Generated Holograms ^
  // //by W.J.Dallas all we have to do is :
  // // 1) Load the object (image in our case)
  // // 2) Apply random phase diffuser
  // // 3) Inverse fourier transform
  {  // returning the result
  // // compute the magnitude and switch to logarithmic scale
    merge(planes, 2, g_ROI);
    return g_ROI;
  }
}

void eit_hologram::qshow(cv::Mat& in, std::string name){
  if(in.channels()==1){
    imshow("Qshow " + name, in);
  }else if(in.channels()==2){
    cv::Mat planes_Hw[] = {cv::Mat::zeros(in.size(), CV_32F),
			   cv::Mat::zeros(in.size(), CV_32F)}; //Filled with real part of the image
    split(in,planes_Hw);

    imshow("Qshow re "+name, planes_Hw[0]);
    imshow("Qshow im "+name, planes_Hw[1]);
  }
}

cv::Mat idft_one(Mat inputdata){
  //Receives DFT complex input data 
  //returns the original image
  inputdata.convertTo(inputdata, CV_32FC1);
  Mat inverse = inputdata;
  idft(inverse,inverse);

  Mat planes[] = {Mat_<float>(inputdata), Mat::zeros(inputdata.size(), inputdata.type())};
  split(inverse,planes);
  magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
  Mat magI = planes[0];
  magI += Scalar::all(1);                    // switch to exp scale
  //exp(magI,magI); //Supposedly, input data should be logarithm'd
  normalize(magI,magI,0,1,CV_MINMAX);
  return magI;
}

cv::Mat idft_two(Mat magI){
  //Receives DFT complex input data 
  //returns the original image
  magI.convertTo(magI,CV_32FC1);
  cv::dft(magI, magI, cv::DFT_INVERSE|cv::DFT_REAL_OUTPUT);
  normalize(magI,magI,0,1,CV_MINMAX);
  return magI;
}



