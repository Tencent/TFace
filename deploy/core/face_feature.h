#ifndef _FACE_FEATURE_H_
#define _FACE_FEATURE_H_

#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <tnn/core/tnn.h>
#include <tnn/utils/blob_converter.h>
#include <tnn/utils/dims_vector_utils.h>
#include <tnn/utils/mat_utils.h>

class TNNInstance{
 public:
    TNNInstance() {}
    ~TNNInstance() {}
    TNN_NS::Status Init(const std::string&, const std::string& model);
    TNN_NS::Status Forward(TNN_NS::Mat& input_mat, std::vector<float>& feature);
    TNN_NS::DimsVector GetInputDims(std::string layer_name = "");
    TNN_NS::DimsVector GetOutputDims(std::string layer_name = "");

 private:
    std::shared_ptr<TNN_NS::Instance> instance_ = nullptr;
    std::map<std::string, TNN_NS::DimsVector> input_dims_;
    std::map<std::string, TNN_NS::DimsVector> output_dims_;
    std::map<std::string, std::shared_ptr<TNN_NS::Mat>> output_mats_;
};

#endif  // _FACE_FEATURE_H_