#include <iostream>
#include <fstream>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include "gflags/gflags.h"
#include "face_feature.h"
#include "util.h"
#include "timer.h"

using namespace std;
using namespace cv;

DEFINE_string(img, "./data/brucelee.jpg", "input img");
DEFINE_string(proto, "", "tnn proto path");
DEFINE_string(model, "", "tnn model path");
DEFINE_bool(h, false, "help");

void ShowUsage() {
    printf("    -h     <help>\n");
    printf("    -img   <input img>\n");
    printf("    -proto <tnn proto path>\n");
    printf("    -model <tnn model path>\n");
}

bool ParseAndCheckCommandLine(int argc, char* argv[]) {
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        ShowUsage();
        return false;
    }
    if (FLAGS_proto.empty()) {
        printf("tnn proto path is not set\n");
        ShowUsage();
        return false;
    }

    if (FLAGS_model.empty()) {
        printf("tnn model path is not set\n");
        ShowUsage();
        return false;
    }
    return true;
}

int main(int argc, char* argv[]) {
    if (!ParseAndCheckCommandLine(argc, argv))
        return -1;

    Mat img = imread(FLAGS_img);
    cvtColor(img, img, cv::COLOR_BGR2RGB);
    printf("image width: %d, height: %d\n", img.cols, img.rows);
    string proto_path = FLAGS_proto;
    string model_path = FLAGS_model;

    string proto, model;
    int ret = 0;
    ret = ReadFile(proto_path, proto);
    if (ret) return ret;
    ret = ReadFile(model_path, model);
    if (ret) return ret;

    TNNInstance* instance = new TNNInstance();
    TNN_NS::Status status;
    status = instance->Init(proto, model);
    if (status != 0) {
        return status;
    }
    TNN_NS::DimsVector input_dims = instance->GetInputDims();
    TNN_NS::Mat input_mat = TNN_NS::Mat(TNN_NS::DEVICE_NAIVE, TNN_NS::N8UC3, input_dims, img.data);

    vector<float> feature;
    status = instance->Forward(input_mat, feature);
    if (status != 0) {
        return status;
    }
    for (float i : feature)
        printf("%f ", i);
    printf("\n");

    Timer timer("feature");

    for (size_t i = 0; i < 100; i++) {
        timer.Start();
        status = instance->Forward(input_mat, feature);
        if (status != 0)
            return status;
        timer.Stop();
    }
    timer.Print();

    return 0;
}
