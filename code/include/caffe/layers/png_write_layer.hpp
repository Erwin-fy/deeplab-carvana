#ifndef CAFFE_PNG_WRITE_LAYER_HPP_
#define CAFFE_PNG_WRITE_LAYER_HPP_

#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/*
  MatWriteLayer
*/
template <typename Dtype>
class PNGWriteLayer : public Layer<Dtype> {
 public:
  explicit PNGWriteLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "PNGWrite"; }
  virtual inline int ExactNumTopBlobs() const { return 0; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int iter_;
  int period_;
  string prefix_;
  vector<string> fnames_;
};

}  // namespace caffe

#endif  // CAFFE_PNG_WRITE_LAYER_HPP_
