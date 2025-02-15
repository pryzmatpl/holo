#include "optimize.hpp"

using namespace std;

typedef mat_rows uchar*; 
typedef mat_row uchar;



Mat& basic_operation(Mat& input, Mat& input_before, Mat& desired){
  Mat returnee;
  returnee.clone(input); //Add the used mat header
  //TODO: Add sanity checks

  int channels = input.channels();
  int rows     = input.rows;
  int cols     = input.cols*channels;

  for(int row = 0; row < rows; row++){
    mat_rows t       =        input.ptr<mat_row>(row);
    mat_rows tminus1 = input_before.ptr<mat_row>(row);
    mat_rows r       =     returnee.ptr<mat_row>(row);
    mat_rows d       =      desired.ptr<mat_row>(row);

    for(int col = 0; col < cols; col++){
      if((col > 1) && (col < cols)){
	//TODO: add sanity checks
	r[col] = abs( t[col] - ( tminus1[col] - d[col] ) );
      }
    }
  }
}


