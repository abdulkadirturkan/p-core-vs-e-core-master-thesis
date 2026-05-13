#include <vector>
#pragma once

template<typename T>
class Tensor {
public:
    std::vector<std::vector<std::vector<T>>> data;
    int channels, height, width;

    Tensor(int c, int h, int w) : channels(c), height(h), width(w) {
        data.resize(c, std::vector<std::vector<T>>(h, std::vector<T>(w, 0)));
    }
};