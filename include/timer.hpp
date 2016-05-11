#pragma once

#include <chrono>
using namespace std::chrono;

class Timer{
public:
    void start(){
        t1 = high_resolution_clock::now();
    }
    double stop(){
        t2 = high_resolution_clock::now();
        time_span = duration_cast<duration<double>>(t2 - t1);
        return time_span.count(); // in seconds
    }
    double elapsedTime(){
        t2 = high_resolution_clock::now();
        time_span = duration_cast<duration<double>>(t2 - t1);
        return time_span.count(); // in seconds
    }
    double elapsedTimeMilliSeconds(){
        t2 = high_resolution_clock::now();
        time_span = duration_cast<milliseconds>(t2 - t1);
        return time_span.count(); // in milliseconds
    }
    double elapsedTimeMicroSeconds(){
        t2 = high_resolution_clock::now();
        time_span = duration_cast<microseconds>(t2 - t1);
        return time_span.count(); // in microseconds
    }
private:
    high_resolution_clock::time_point t1;
    high_resolution_clock::time_point t2;
    duration<double> time_span;
};

