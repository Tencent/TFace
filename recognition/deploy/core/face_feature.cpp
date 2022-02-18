#include <cstdio>
#include <cstdlib>
#include "face_feature.h"
#include "util.h"

using namespace std;
using namespace cv;

TNN_NS::Status TNNInstance::Init(const std::string& proto, const std::string& model) {
    TNN_NS::ModelConfig config;
    config.model_type = TNN_NS::MODEL_TYPE_TNN;
    config.params.push_back(proto);
    config.params.push_back(model);
    TNN_NS::TNN net;
    TNN_NS::Status status = net.Init(config);
    if (!CheckResult("init tnn", status)) {
        return status;
    }

    TNN_NS::NetworkConfig network_config;
    network_config.precision = TNN_NS::PRECISION_HIGH;
    network_config.device_type = TNN_NS::DEVICE_X86;
    #ifdef _OPENVINO_
        network_config.network_type = TNN_NS::NETWORK_TYPE_OPENVINO;
    #endif
    instance_ = net.CreateInst(network_config, status);
    if (!CheckResult("create instance", status)) {
        return status;
    }
    instance_->SetCpuNumThreads(1);
    TNN_NS::BlobMap input_blobs;
    instance_->GetAllInputBlobs(input_blobs);
    for (auto iter : input_blobs) {
        TNN_NS::DimsVector input_dims = iter.second->GetBlobDesc().dims;
        input_dims_[iter.first] = input_dims;
        #ifdef _LOG_
            fprintf(stdout, "input layer: %s, dims: %d/%d/%d/%d\n", \
                iter.first.c_str(), input_dims[0], input_dims[1], input_dims[2], input_dims[3]);
        #endif
    }

    TNN_NS::BlobMap output_blobs;
    instance_->GetAllOutputBlobs(output_blobs);
    for (auto iter  : output_blobs) {
        TNN_NS::DimsVector output_dims = iter.second->GetBlobDesc().dims;
        output_dims_[iter.first] = output_dims;
        #ifdef _LOG_
            fprintf(stdout, "output layer: %s, dims: %d/%d\n", iter.first.c_str(), output_dims[0], output_dims[1]);
        #endif
    }
    return TNN_NS::TNN_OK;
}


TNN_NS::Status TNNInstance::Forward(TNN_NS::Mat& input_mat, vector<float>& feature) {
    if (!instance_) {
        return TNN_NS::Status(-10, "TNN instance not init");
    }
    TNN_NS::Status status;
    TNN_NS::BlobMap input_blob_map, output_blob_map;
    instance_->GetAllInputBlobs(input_blob_map);
    instance_->GetAllOutputBlobs(output_blob_map);

    // input mat and converter
    TNN_NS::MatConvertParam input_convert_param;
    input_convert_param.scale = {1.0f / 127.5, 1.0f / 127.5, 1.0f / 127.5, 1.0f / 127.5};
    input_convert_param.bias = {-1.0, -1.0, -1.0, -1.0};
    TNN_NS::BlobConverter input_blob_convert(input_blob_map.begin()->second);
    status = input_blob_convert.ConvertFromMat(input_mat, input_convert_param, nullptr);
    if (!CheckResult("convert from mat", status)) {
        return status;
    }

    // output mat and converter
    TNN_NS::MatConvertParam output_convert_param;
    TNN_NS::DimsVector dims = GetOutputDims();
    auto output_mat = std::make_shared<TNN_NS::Mat>(TNN_NS::DEVICE_NAIVE, TNN_NS::NCHW_FLOAT, dims);
    TNN_NS::BlobConverter output_blob_convert(output_blob_map.begin()->second);

    // forward
    status = instance_->Forward();
    if (!CheckResult("forward", status)) {
        return status;
    }

    // get output
    status = output_blob_convert.ConvertToMat(*output_mat, output_convert_param, nullptr);
    if (!CheckResult("convert to mat", status)) {
        return status;
    }
    int m_feature_length = TNN_NS::DimsVectorUtils::Count(dims);
    float* output_data = static_cast<float *>(output_mat.get()->GetData());
    vector<float> data(output_data, output_data + m_feature_length);
    vector<float> norm = l2_normlize(data);
    feature.clear();
    feature.resize(m_feature_length);
    memcpy(feature.data(), norm.data(), m_feature_length * sizeof(float));
    return TNN_NS::TNN_OK;
}

TNN_NS::DimsVector TNNInstance::GetInputDims(std::string layer_name) {
    if (layer_name == "") {
        layer_name = input_dims_.begin()->first;
    }
    TNN_NS::DimsVector dims = input_dims_[layer_name];
    #ifdef _LOG_
        fprintf(stdout, "mat dims from layer: %s, dims: %d/%d/%d/%d\n", \
            layer_name.c_str(), dims[0], dims[1], dims[2], dims[3]);
    #endif
    return dims;
}

TNN_NS::DimsVector TNNInstance::GetOutputDims(std::string layer_name) {
    if (layer_name == "") {
        layer_name = output_dims_.begin()->first;
    }
    TNN_NS::DimsVector dims = output_dims_[layer_name];
    #ifdef _LOG_
        fprintf(stdout, "mat dims from layer: %s, dims: %d/%d\n", layer_name.c_str(), dims[0], dims[1]);
    #endif
    return dims;
}