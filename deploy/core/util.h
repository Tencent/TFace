#ifndef _UTIL_H_
#define _UTIL_H_

#include <iostream>
#include <fstream>

#include <tnn/core/tnn.h>

inline bool CheckResult(std::string desc, TNN_NS::Status result) {
    if (result != 0) {
        #ifdef _LOG_
            fprintf(stderr, "[%s] failed: %s \n", desc.c_str(), result.description().c_str());
        #endif
        return false;
    } else {
        #ifdef _LOG_
            fprintf(stdout, "[%s] success! \n", desc.c_str());
        #endif
        return true;
    }
}

inline float norm(std::vector<float> &fin) {
    float sum = 0.f;
    for (float v : fin) {
        sum += v * v;
    }
    return sqrt(sum);
}

inline std::vector<float> l2_normlize(std::vector<float> & fin) {
    float div = norm(fin);
    std::vector<float> vout;
    for (auto it = fin.begin(); it != fin.end(); it++) {
        float value = (*it) / div;
        vout.push_back(value);
    }
    return vout;
}

inline int ReadFile(const std::string& file_path, std::string &data) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file) {
        fprintf(stderr, ("file %s not exist\n", file_path.c_str()));
        return -1;
    }

    file.seekg(0, std::ios::end);
    std::streampos fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    if (fileSize <= 0) {
        fprintf(stderr, ("file %s is empty\n", file_path.c_str()));
        return -2;
    }

    data = std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    return 0;
}

#endif