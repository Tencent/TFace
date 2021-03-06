#ifndef _TIMER_H_
#define _TIMER_H_

#include <cstdio>
#include <cfloat>
#include <cmath>
#include <chrono>
#include <string>

using std::chrono::time_point;
using std::chrono::system_clock;
using std::chrono::duration_cast;
using std::chrono::microseconds;

class Timer {
 public:
    explicit Timer(std::string timer_info) {
        timer_info_ = timer_info;
        Reset();
    }
    void Start() {
        start_ = system_clock::now();
    }
    void Stop() {
        stop_ = system_clock::now();
        float delta = duration_cast<microseconds>(stop_ - start_).count() / 1000.0f;
        min_ = static_cast<float>(fmin(min_, delta));
        max_ = static_cast<float>(fmax(max_, delta));
        sum_ += delta;
        count_++;
    }
    void Reset() {
        min_ = FLT_MAX;
        max_ = FLT_MIN;
        sum_ = 0.0f;
        count_ = 0;
        stop_ = start_ = system_clock::now();
    }
    void Print() {
        char min_str[16];
        snprintf(min_str, 16, "%6.3f", min_);
        char max_str[16];
        snprintf(max_str, 16, "%6.3f", max_);
        char avg_str[16];
        snprintf(avg_str, 16, "%6.3f", sum_ / static_cast<float>(count_));
        printf("%-15s cost: min = %-8s ms  |  max = %-8s ms  |  avg = %-8s ms \n", timer_info_.c_str(),
            min_str, max_str, avg_str);
    }

 private:
    float min_;
    float max_;
    float sum_;
    std::string timer_info_;
    time_point<system_clock> start_;
    time_point<system_clock> stop_;
    int count_;
};

#endif  // _TIMER_H_