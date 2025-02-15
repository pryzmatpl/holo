#include "../include/filters.hpp"
#include "../include/fft.hpp"

//TODO

using namespace std;
using namespace cv;
extern float wavelength; //TODO: I shiver upon this global, already causing nausea

cv::Mat eit_hologram::holoeye_filter_c1(cv::Mat& input, 
					std::function<void (cv::Mat&)> filter){
  //Filter should be CV32_FC1
  Mat returnee(input);
  filter(returnee);
  return returnee;
}

holo_filter filter_checker = [](cv::Mat& in){
  Mat returnee[2] = { Mat_<float>(in.size(), in.type()),
		      Mat::zeros(in.size(), in.type())};

  if(in.channels() == 2){
    split(in,returnee);
  }else{
    returnee[0] = in;
  }

  int rowmid = in.rows/2+1;
  int colmid = in.cols/2+1;

  for(int row=0; row<in.rows; row++){
    for(int col=0; col<in.cols; col++){
      //channel check
      if( (abs(row-rowmid)+abs(col-colmid)) < 70 ){
	if((row+col)%2==0){
	  if(in.channels() == 2){
	    float& re = returnee[0].at<float>(row,col);
	    float& im = returnee[1].at<float>(row,col);
	    
	    re = 0.0f;
	    im = 0.0f;
	  }else{
	    float& val = returnee[0].at<float>(row,col);
	    val = 0.0f;
	  }
	}else{
	  if(in.channels() == 2){
	    float& re = returnee[0].at<float>(row,col);
	    float& im = returnee[1].at<float>(row,col);

	    re += 1.0f;
	    im -= 1.0f;
	  }else{
	    float& val = returnee[0].at<float>(row,col);
	    val += 2.0f;
	  }
	}
      }
    }
  }

  if(in.channels() == 2){
    merge(returnee,2,in);
  }else{
    in = returnee[0];
  }
};

holo_filter filter_inverse_fourier_kernel_one = [](cv::Mat& in){
  //filter taken from denis et al - numerical suppression of the twin image in in-line holo...
  //normalize(in,in,0,1,CV_MINMAX);
  Mat returnee[2] = { Mat_<float>(in.size(), in.type()),
		      Mat::zeros(in.size(), in.type())};

  if(in.channels() == 2){
    split(in,returnee);
  }else{
    returnee[0] = in;
  }

  int rowmid = in.rows/2+1;
  int colmid = in.cols/2+1;
  
  for(int row = 0; row < in.rows; row++){
    for(int col = 0; col < in.cols; col++){
      float& pxl_re = returnee[0].at<float>(row,col);
      float& pxl_im = returnee[1].at<float>(row,col);

      std::complex<float> val_filt = 1.0f/(1.0f+exp(neg_I*2.0f*pi*wavelength*glob_distance*(float)( (row-rowmid)*(row-rowmid) + (col-colmid)*(col-colmid) ) ) );
      std::complex<float> val_img = std::complex<float>(pxl_re,pxl_im);
      val_img*=val_filt;
      pxl_re=val_img.real();
      pxl_im=val_img.imag();
    }
  }

  if(in.channels() == 2){
    merge(returnee,2,in);
  }else{
    in = returnee[0];
  }
};

holo_filter filter_rpn = [](cv::Mat& in){
  Mat returnee[2] = { Mat_<float>(in.size(), in.type()),
		      Mat::zeros(in.size(), in.type())};
  split(in, returnee);

  boost::mt19937 rng; 
  boost::normal_distribution<> nd(0.0,1.0);
  boost::variate_generator<boost::mt19937&, 
                           boost::normal_distribution<> > var_nor(rng, nd);

  for(int row = 0; row < in.rows; row++){
    for(int col = 0; col < in.cols; col++){
      float& pxl_re = returnee[0].at<float>(row,col);
      float& pxl_im = returnee[1].at<float>(row,col);
      float rnd_phase_rad = var_nor(); //first generation of phase
      
      std::complex<float> val = std::polar<float>(in.at<float>(row,col),0);
      std::complex<float> phasevalue(exp(neg_I*rnd_phase_rad));
      
      val += phasevalue;
      
      pxl_re = val.real();
      pxl_im = val.imag();
    }
  }

  merge(returnee, 2, in);
};

holo_filter filter_spherical = [](cv::Mat& in){
  //taken from Zhang,Hao - Eliminiation of a zero-order beam induced by pixelated SLM
  Mat returnee[2] = { Mat_<float>(in.size(), in.type()),
		      Mat::zeros(in.size(), in.type())};
  if(in.channels() == 2){
    split(in,returnee);
  }else{
    returnee[0] = in;
  }
  
  int rowmid = in.rows/2+1;
  int colmid = in.cols/2+1;
  
  for(int row = 0; row < in.rows; row++){
    for(int col = 0; col < in.cols; col++){
      float& re = returnee[0].at<float>(row,col);
      float& im = returnee[1].at<float>(row,col);

      complex<float> pinfo(re,im);
      complex<float> paddition = polar(1.0f, ((2*pi/wavelength)/glob_distance)*
				       ((row - rowmid)*(row-rowmid) +
					(col - colmid)*(col-colmid)));

      complex<float> ret = polar(sqrt(norm(pinfo)), arg(paddition) + arg(pinfo));
      
      re = ret.real();
      im = ret.imag();
    }
  }
};

holo_filter filter_rpn_ocv = [](cv::Mat& in){
  randn(in,128,30); //TODO: Magic numbers here ;)
};

holo_filter filter_eit_lpl = [](cv::Mat& in){
  //Testing an integral image formation
  //of linear phase loading
  Mat ret(in.cols+1, in.rows+1, in.depth()); //prepare for integral function
  Mat ret_sq(in.rows+1, in.cols+1, in.depth()); //prepare for integral function
  Mat ret_tilt(in.rows+1, in.cols+1, in.depth()); //prepare for integral function

  integral(in,
	   ret,
	   ret_sq,
	   ret_tilt,
	   in.depth());

  in = Mat(ret_sq, Rect(1, 1, in.cols, in.rows));

  return in;
};


holo_filter filter_linear_phase_load = [](cv::Mat& in){
//taken from Zhang,Hao - Eliminiation of a zero-order beam induced by pixelated SLM
  Mat returnee[2] = { Mat_<float>(in.size(), in.type()),
		      Mat::zeros(in.size(), in.type())};
  
  if(in.channels() == 2){ split(in,returnee); }
  else{ returnee[0] = in; }

  int rowmid = in.rows/2+1;
  int colmid = in.cols/2+1;
  float oblique = 20.0f;
  
  double minVal; 
  double maxVal; 

  float factor = 1.0f/256.0f; //TODO: get rid of magic constant
  
  for(int row = 0; row < in.rows; row++){
    for(int col = 0; col < in.cols; col++){
      float& re = returnee[0].at<float>(row,col);
      float& im = returnee[1].at<float>(row,col);

      complex<float> pinfo(re,im);
      complex<float> ret;
      
      float aFactor = ((float)colmid/(float)rowmid);
      float loadedphase = aFactor*(row-rowmid+col-colmid);
      ret = polar(sqrt(norm(pinfo)), loadedphase + factor*arg(pinfo));
      
      re = ret.real();
      im = ret.imag();
    }
  }

  merge(returnee,2,in);
};

holo_filter filter_real_laplacian = [](cv::Mat& in){
  GaussianBlur(in,in,Size(3,3),0,0,BORDER_DEFAULT);
  Laplacian(in,in,in.type(),3,1,0,BORDER_DEFAULT);
};


holo_filter filter_clear_center = [](cv::Mat& in){
  Mat returnee[2] = { Mat_<float>(in.size(), in.type()),
		      Mat::zeros(in.size(), in.type())};
  split(in, returnee);

  int rowmid = returnee[0].rows/2+1;
  int colmid = returnee[0].cols/2+1;
  
  int border = (int)(0.1*returnee[0].rows);

  for(int row = rowmid - border; row < rowmid + border; row++){
    for(int col = colmid - border; col < colmid + border; col++){
      float& re = returnee[0].at<float>(row,col);
      float& im = returnee[1].at<float>(row,col);
      
      if(abs((row-rowmid)+(col-colmid)) < border){
	re = 1.0f;
	im = sqrt((row-rowmid)*(row-rowmid)*wavelength*wavelength+
		  (col-colmid)*(col-colmid)*wavelength*wavelength);
      }
    }
  }

  merge(returnee, 2, in);
};

//TODO: create 2D iterator for opencv mat

holo_filter filter_sinc = [](cv::Mat& in){
  int rowmid = in.rows/2+1;
  int colmid = in.cols/2+1;

  for(row aRow = 0; aRow < in.rows; aRow++){
    for(col aCol = 0; aCol < in.cols; aCol++){
      float flx = boost::math::sinc_pi<float>( ((2*pi) / (wavelength*glob_distance) )  *
					       ((float)( (aRow-rowmid)*(aRow-rowmid) ) * (holoeye_delta_x)*(holoeye_delta_x) +
						(float)( (aCol-colmid)*(aCol-colmid) ) * (holoeye_delta_x)*(holoeye_delta_x) ));

      complex<float> sinc_arg = std::polar(1.0f, flx);
      Vec3f& pxl = in.at<Vec3f>(aRow,aCol);

      complex<float> value(pxl[0], ([&]()->float{ if(in.channels()==1) return 0.0f; else return pxl[1]; }() ));
      value *= sinc_arg;

      pxl[0] = value.real();
      pxl[1] = [&]()->float{ if(in.channels()==1) return 0.0f; else return value.imag(); }();
    }    
  }
};

holo_filter filter_remavg = [](cv::Mat& in){
  Scalar_<float> mean_ = mean(in);
  
  for(int row=0; row < in.rows; row++){
    for(int col=0; col < in.cols; col++){
      Scalar_<float> fl = in.at<Scalar_<float>>(row,col);
      fl -= mean_;
    }
  }
};

cv::Mat eit_hologram::holoeye_filter(cv::Mat& input, 
				     std::string fil_type){
  try{
    if(fil_type == "REMAVG"){ 
      cout<<"filter_remavg <";
      holoeye_filter_c1(input,filter_remavg);
    }else if(fil_type == "SINC"){
      cout<<"filter_sinc <";
      holoeye_filter_c1(input,filter_sinc);
    }else if(fil_type == "LAPLACIAN"){
      cout<<"filter_real_laplacian <";
      holoeye_filter_c1(input,filter_real_laplacian);
    }else if(fil_type == "SPHERICAL"){
      cout<<"filter_spherical <";
      holoeye_filter_c1(input,filter_spherical);
    }else if(fil_type == "RPN"){
      cout<<"filter_rpn <";
      holoeye_filter_c1(input,filter_rpn);
    }else if(fil_type == "LINEAR"){
      cout<<"filter_linear <";
      holoeye_filter_c1(input,filter_linear_phase_load);
    }else if(fil_type == "EIT_LPL"){
      cout<<"filter_eit_lpl <";
      holoeye_filter_c1(input,filter_eit_lpl);
    }else if(fil_type == "RPNOCV"){
      cout<<"filter_rpn_ocv <";
      holoeye_filter_c1(input,filter_rpn_ocv);
    }else if(fil_type == "TWINREM"){
      cout<<"filter_inverse_fourier_kernel_one <";
      holoeye_filter_c1(input,filter_inverse_fourier_kernel_one);
    }else if(fil_type == "CLEARCENTER"){
      cout<<"filter_clear_center <";
      holoeye_filter_c1(input,filter_clear_center);
    }else if(fil_type == "NONE"){
      cout<<"(filimage=NONE)";
    }else if(fil_type == "CHECKER"){
      cout<<"(filimage=CHECKER)";
      holoeye_filter_c1(input,filter_checker);
    }else{};

    cout<<flush;
    return input;
  }catch(std::exception &e){
    cout<<"Exception in holoeye_filter(Mat,Mat,string) -"<<e.what()<<"\n";
  }
}


cv::Mat eit_hologram::holoeye_filter_spatial(cv::Mat& input, cv::Mat& spatial_filter){
  //We get the spatial frequencies of phase and remove certain blocks
  Mat returnee[2] = { Mat_<float>(input.size(), input.type()),
		      Mat::zeros(input.size(), input.type())};
  input.copyTo(returnee[0]); // physically copy to returnee


  Mat filter[2] = { Mat_<float>(input.size(), input.type()),
		    Mat::zeros(input.size(), input.type())};

  resize(spatial_filter, filter[0], input.size());
  filter[0].convertTo(filter[0], CV_32FC2);

  Mat ret_dft(input.size(), CV_32FC2);
  Mat filter_dft(input.size(), CV_32FC2);

  //Assume CV32_FC2 for everything
  merge(returnee, 2, ret_dft);
  dft(ret_dft, ret_dft);
  
  merge(filter, 2, filter_dft);
  dft(filter_dft, filter_dft);

  split(ret_dft, returnee);
  split(filter_dft,filter);
  
  try{
    for(int row = 0; row < input.rows; row++){
      for(int col = 0; col < input.cols; col++){
	auto final_pxl_value = complex<float>(returnee[0].at<float>(row,col),
					      returnee[1].at<float>(row,col)); //the first imaginary value
	auto filter_pxl_value = complex<float>(filter[0].at<float>(row,col),
					       filter[1].at<float>(row,col)); //the second imag of the filter

	final_pxl_value *= filter_pxl_value; //pointwise multiplication
	
	float& ret_real = returnee[0].at<float>(row,col);
	float& ret_imag = returnee[1].at<float>(row,col);

	ret_real = final_pxl_value.real();
	ret_imag = final_pxl_value.imag();
      }
    }

    merge(returnee, 2, ret_dft);
    dft(ret_dft, ret_dft, DFT_INVERSE);
    split(ret_dft, returnee);
    phase(returnee[0], returnee[1], returnee[0]);

  }catch(cv::Exception& e){
    cout<<"Official error:"<<e.what()<<endl;
    exit(-1);
  }

  return returnee[0];
}

