#include "../main.hpp"
using namespace std;
using namespace cv;
namespace po = boost::program_options;

float glob_distance;
int zoom;
const int zoommax=100;
bool flipzero;

typedef boost::char_separator<char> separator_type;

std::string get_file_contents(const char *filename)
{
  //my second favorite lambda today
  string contents="",temp;
  fstream input(filename,std::ios::in);
  if(input.is_open()){
    while(std::getline(input,temp)){
      temp.erase(std::remove_if(temp.begin(),
				temp.end(),
				[](char c){ return (c=='\r' || c=='\t' || c==' ' || c=='\n'); }),
		 temp.end());
      contents+=temp;
    }
  }
  return contents;
}

int main(int argc, char** argv){
  const char* fname = (argc>=1)? argv[1] : "_logo.jpg";
  
  namedWindow("Output - intertwined",CV_WINDOW_NORMAL);

  po::options_description desc("Options:");
  desc.add_options()
    ("help","About the program")
    ("maskfile", po::value<string>(), "File with the pixel list")
    ("frequency", po::value<float>(), "How many times to switch the images per second");
  po::variables_map vm;
  po::store(po::parse_command_line(argc,argv,desc),vm);
  po::notify(vm);

  string maskfile;
  int frequency;
  { //Display the help
    if(vm.count("help")){
      cout<<desc<<"\n";
      return 1;
    }
  }
  { //Maskifle
    if(vm.count("maskfile")){
      maskfile = vm["maskfile"].as<string>();
    }else{
    }
  }
  { 
    if(vm.count("frequency")){
      frequency = vm["frequency"].as<float>();
    }else{
      frequency = 10.0f;
    }
  }

  //Prepare the image data structure to push them out
  std::vector<Mat> images;
  string pixel_setup = get_file_contents(maskfile.c_str());
  boost::tokenizer<separator_type> pixels(pixel_setup,separator_type(";"));

  for(boost::tokenizer<separator_type>::iterator it = pixels.begin();
      it != pixels.end();
      it++){
    Mat input_image;
    input_image = imread(*it);
    images.push_back(input_image);
  }
  
  namedWindow("Alternating",CV_WINDOW_NORMAL);
  imshow("Alternating",images[0]);
  
  int ms_of =  (int)1000.0f/frequency;
  cout<<ms_of<<"\n";

  bool tiled_fs;
  
  int idx=0;
  int images_sz = images.size();
  
  while(int x=waitKey(ms_of)){
    try{
      idx++;
      idx %= images_sz;
      imshow("Alternating", images[idx]);
    }catch(std::exception &e){
      cout <<"Exception: "<< e.what() <<"\n";
    }
    
    if(x==102){
      if(!tiled_fs){
	setWindowProperty("Alternating", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
	//setWindowProperty("tiled_conv",CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
	tiled_fs=true;
      }else{
	tiled_fs=false;
	setWindowProperty("Alternating", CV_WND_PROP_FULLSCREEN, CV_WINDOW_NORMAL);
	//setWindowProperty("tiled_conv",CV_WND_PROP_FULLSCREEN, CV_WINDOW_NORMAL);
      }
    }
  }

}
