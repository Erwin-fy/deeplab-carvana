#include <stdint.h>

#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "matio.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace cv;

namespace caffe {

template <typename Dtype>
void WriteBlobToPNG(const char *fname, bool write_diff,
    Blob<Dtype>* blob) {
  size_t dims[4];
  dims[0] = blob->width();
  dims[1] = blob->height();
  dims[2] = blob->channels();
  dims[3] = blob->num();
  
  Mat src(961, 1281, CV_8U);
  
  for (int h=0; h < blob->height(); ++h) {
    for (int w=0; w < blob->width(); ++w) {
      src.at<uchar>(h, w) = blob->data_at(0,0,h,w) * 255;
    }
  }

  Mat dst;
  resize(src, dst, Size(1918, 1280));
  vector<int>compression_params;    
  compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION); //PNG格式图片的压缩级别    
  compression_params.push_back(9);    
  imwrite(fname, dst, compression_params);   
    
}


template void WriteBlobToPNG<float>(const char*, bool, Blob<float>*);
template void WriteBlobToPNG<double>(const char*, bool, Blob<double>*);
template void WriteBlobToPNG<int>(const char*, bool, Blob<int>*);
template void WriteBlobToPNG<unsigned int>(const char*, bool, Blob<unsigned int>*);

}
