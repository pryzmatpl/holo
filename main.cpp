#include "main.hpp"

using namespace std;
using namespace cv;
namespace po = boost::program_options;

float glob_distance=1;
float wavelength=632.8e-9;
int zoom=1;
const int zoommax=1000;
Mat backup__;
extern float frft_angle;

void onzoom(int trackposition, void* userdata){
  glob_distance = ((float)trackposition/(float)zoommax)*(float)zoommax;
  eit_hologram eit_holo;
  Mat& indata = reinterpret_cast<Mat&>(*userdata);

  indata.copyTo(backup__);
  backup__ = eit_holo.holoeye_filter(backup__,"SINC");
  backup__ = eit_holo.tile_to_fhd(backup__);
  imshow("Output - tiled", backup__);
};

inline void quickImg(string name, Mat& input){
  if(input.channels() == 2){
    Mat news[2];
    split(input,news);
    phase(news[0],news[1],news[0]);
    normalize(news[0],news[0],0,1,CV_MINMAX);
    imshow(name.c_str(), news[0]);
  }else{
    imshow(name.c_str(), input);
  }
}

//TODO: use opencv's filtering infrastructure
//TODO: Gernchberg-Saxton?
int main(int argc, char** argv){
  const char* fname = (argc>=1)? argv[1] : "_logo.jpg";
  //Working transforms:
  //dft
  //rpn
  //Fresnel as convolution

  //Working reference phases:
  //ref_wavefront_holoeye_two
  //holoeye_chirp for Fresnel
  //Best effects:
  //Fresnel + remove complex average

  float distance;
  string filename;
  string filename_spatial_filter;
  string holotype; //RPN, or Fresnel, or FFT
  string refwave; //Methods from filters.cpp v
  string filimage; //Methods from filters.cpp v
  string filcimage;  
  string filafft;
  string filout;
  string chirp;
  bool gentwo;
  bool flipzero;
  bool gerch_on;
  
  namedWindow("Output - tiled",CV_WINDOW_NORMAL);
  eit_hologram eit_holo;
  gerch Gerch;
  Mat adft_data,adft_data_2;
  Mat fresnel;
  Mat fresn_conv;
  Mat spatial_filter;
  Mat padded;                          //expand input image to optimal size
  Mat g_ROI;

  po::options_description desc("Options:");
  desc.add_options()
    ("help","About the program")
    ("image", po::value<string>(), "File that will be changed into hologram <string>")
    ("distance" , po::value<float>(), "Distance from SLM to hologram <float>")
    ("type", po::value<string>(), "Type of hologram <string> (FRESNEL, RPN, RPNOCV, FFT, FRECONV) ")
    ("gerch" , po::value<bool>(), "Use Gerchberg-Saxton algorithm for the transform")
    ("refwave", po::value<string>(), "What reference wavefront type, in <string> (FLAT, REF, CHIRP, RAYLEIGH) ")
    ("filimage", po::value<string>(), "What filtering of the real input image, type: <string> (REMAVG, SINC, LAPLACIAN, RPN, CLEARCENTER, TWINREM)")
    ("filcimage", po::value<string>(), "What filtering for the complex input image w/ reference wavefront phase, type: <string>")
    ("filafft", po::value<string>(), "What filtering of the complex FFT of the input image, type: <string>")
    ("filout", po::value<string>(), "What filtering of the output, type: <string>")
    ("wavelength", po::value<float>(), "What wavelength is required? default: He-Ne, <float>")
    ("gentwo", po::value<bool>(), "Generate two images for testing purposes w/ different refwaves <bool>")
    ("frft_angle", po::value<float>(), "Fractional Fourier transform angle")
    ("chirp", po::value<string>(), "Add a chirp function after the FFT of the complex input (real image + ref phase) and pointwise multiply, type:<string>");
  po::variables_map vm;

  //TODO:
  //Pass the distance to the appropriate reference wave functions and such
  //Pass all the required Params through the computation pipeline
  //CHECK THAT ALL THE REFEERNCES ARE WORKING
  //Take out the Sommerfeld-Rayleigh function out of the fresnel-conv for future use
  po::store(po::parse_command_line(argc,argv,desc),vm);
  po::notify(vm);

  { //Display the help
    if(vm.count("help")){
      cout<<desc<<"\n";
      return 1;
    }
  }

  { //Load the image 
    if(vm.count("image")){
      filename = vm["image"].as<string>();
    }else{
      filename = "_logo.jpg";
    }
  }
  
  { //Set the distance
    if(vm.count("distance")){
      distance = vm["distance"].as<float>();
      glob_distance = distance;
    }else{
      distance = 1.0f;
      glob_distance = distance;
    }
  }
  
  { //Set the distance
    if(vm.count("frft_angle")){
      frft_angle = vm["frft_angle"].as<float>();
    }else{
      frft_angle = pi/2;
    }
  }
  
  { //Load the image
    std::mutex wl_mutex;
    wl_mutex.lock();
    if(vm.count("wavelength")){
      wavelength = vm["wavelength"].as<float>();
    }else{
      wavelength = 632.8e-9;
    }
    wl_mutex.unlock();
  }
  
  { //Set the hologram type
    if(vm.count("type")){
      holotype = vm["type"].as<string>();
    }else{
      holotype = "FRESNEL";
    }
  }

  { //Set the filter for the input image
    if(vm.count("filimage")){
      filimage = vm["filimage"].as<string>();
    }else{
      filimage = "NONE";
    }
    //if not set, then filimage.size() == 0
  }

  { //Set the complex filtering of the complex input
    if(vm.count("filcimage")){
      filcimage = vm["filcimage"].as<string>();
    }
    //if not set, then filcimage.size() == 0
  }

  { //Set the filtering after the FFT
    if(vm.count("filafft")){
      filafft = vm["filafft"].as<string>();
    }
    //if not set, then filafft.size() == 0
  }

  { //Set the reference wavefront
    if(vm.count("refwave")){
      refwave = vm["refwave"].as<string>();
    }
    //if not set, then refware.size() == 0
  }

  { //Set the reference wavefront
    if(vm.count("gerch")){
      gerch_on = vm["gerch"].as<bool>();
    }else{
      gerch_on = true;
    }
    //if not set, then refware.size() == 0
  }
  
  { //Set the filtering of the output image
    if(vm.count("filout")){
       filout = vm["filout"].as<string>();
    }
    //if not set, then filout.size() == 0
  }

  { //Set the chirp function
    if(vm.count("chirp")){
       chirp = vm["chirp"].as<string>();
    }
    //if not set, then chirp.size() == 0;
  }

  { //Create two images with different reference waves
    if(vm.count("gentwo")){
       gentwo = vm["gentwo"].as<bool>();
    }else{
      gentwo = false;
    }
  }
  
  //The region of actual processing of the images
  //First we get a padded image, which we then push through into the processing regime
  g_ROI = imread(filename,CV_LOAD_IMAGE_GRAYSCALE); //Load the real image and save its type
  typetest(g_ROI);
  eit_holo.set_optimal_holosize(g_ROI, padded); //Sets optimal size and converts matrix types
  typetest(padded);

  Mat intensity;
  Mat different_image;
  g_ROI.copyTo(different_image);
  g_ROI.copyTo(intensity);
  /*******************************************************************
   * STEP 1 : Filter the real-valued input image                     *
   ******************************************************************/
  eit_holo.holoeye_filter(padded,
			  filimage); //filter the real image (check if there is a single channel in the filtering functions 

  /*******************************************************************
   * STEP 2 : Add a reference wave to the input                      *
   ******************************************************************/
  eit_holo.holoeye_reference(padded,
			     refwave); //prepare the reference wavefront

  /*******************************************************************
   * STEP 3 : Filter the image with the phase-only reference         *
   ******************************************************************/
  eit_holo.holoeye_filter(padded,
			  filcimage);
  //filter the complex image
  
  /*******************************************************************
   * STEP 4 : Perform the transform fil([fil(img_re)|img_im])        *
   ******************************************************************/
  if(gerch_on){
    Gerch(padded, intensity);
  }
  else
    eit_holo.holoeye_transform(padded,
			       adft_data,
			       holotype); //Perform the transform of the desired type
  /*******************************************************************
   * STEP 5 : Perform fil(xFFT(fil([fil(img_re)|img_im])))           *
   ******************************************************************/
  eit_holo.holoeye_filter(padded,
   			  filafft);
  
  /*******************************************************************
   * STEP 6 : tile(phase(fil(xFFT(fil([fil(img_re)|img_im])))))      *
   ******************************************************************/
  Mat backup;
  padded.copyTo(backup);
  Mat finalres = eit_holo.tile_to_fhd(padded);
  
  /*******************************************************************
   * STEP 7 : fil(tile(phase(fil(xFFT(fil([fil(img_re)|img_im])))))) *
   ******************************************************************/
  eit_holo.holoeye_filter(finalres,
   			  filout);
  zoom = 500;
  padded.copyTo(backup__);

  createTrackbar("Zoom via sinc","Output - tiled",
		 &zoom,
		 zoommax,
		 onzoom,
		 (void*)&finalres);
  Mat _final;
  
  if(gerch_on){
    Mat final = eit_holo.tile_to_fhd(intensity);
    imshow("Output - tiled", final);
    imwrite("./latest-GS.jpg",final);
  }else{
    imshow("Output - tiled", finalres);
    imwrite("./latest.jpg", finalres);
    typetest(intensity);
  }
  
  if(gentwo){
    normalize(finalres,finalres,0,255,CV_MINMAX);
    imwrite("./holobo-1.jpg",finalres);
  }

  cout<<"Wavelength:"<<wavelength<<"\n";
  
  bool tiled_fs=false;
  while(int x=waitKey(30)){
    if(x==102){
      if(!tiled_fs){
	setWindowProperty("Output - tiled", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
	finalres = eit_holo.tile_to_fhd(backup);
	//setWindowProperty("tiled_conv",CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
	tiled_fs=true;
      }else{
	finalres = eit_holo.tile_to_fhd(backup);
	tiled_fs=false;
	setWindowProperty("Output - tiled", CV_WND_PROP_FULLSCREEN, CV_WINDOW_NORMAL);
	//setWindowProperty("tiled_conv",CV_WND_PROP_FULLSCREEN, CV_WINDOW_NORMAL);
      }
    }
    if(x==103){
      
    }
    if(x==97){
      
    }
    if(x==27){
      exit(-1);
    }
  }
}

