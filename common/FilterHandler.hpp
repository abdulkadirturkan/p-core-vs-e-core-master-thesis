#ifndef FILTERHANDLER_HPP
#define FILTERHANDLER_HPP

#include <vector>
#include <string>
#include "Tensor.hpp"
#include <opencv2/opencv.hpp>

enum class FilterType {
    MEAN,
    MEAN1D,
    GAUSSIAN,
    GAUSSIAN1D,
    MEDIAN,
    MODE
};

class FilterHandler {
public:
    static Tensor<float> applyFilter(const Tensor<float>& input, FilterType filterType, int filterSize, int numThreads);

    static Tensor<float> createMeanFilter(int size);
    static Tensor<float> createMeanFilter1D(int filterSize);
    static Tensor<float> createGaussianFilter(int size);
    static Tensor<float> createGaussianFilter1D(int size);
    static Tensor<float> applyHistogramMedianFilter(const Tensor<float>& input, int r);
    static Tensor<float> applyMedianFilter(const Tensor<float>& input, int r);
    static Tensor<float> applyModeFilter(const Tensor<float>& input, int r);

    static Tensor<float> loadImage(const std::string& filePath);
    static Tensor<float> loadRGBImage(const std::string& filePath);

    static bool saveImage(const Tensor<float>& image, const std::string& path);
    static bool saveMultipleChannelsImage(const Tensor<float>& image, const std::string& basePath);
    static bool saveCombinedImage(Tensor<float>& image, const std::string& basePath);

    static std::string getFilterName(FilterType filter);

    static Tensor<float> applyConvolutionHorizontalDivision(const Tensor<float>& input, const Tensor<float>& filter, int numThreads);
    static Tensor<float> applyConvolutionVerticalDivision(const Tensor<float>& input, const Tensor<float>& filter, int numThreads);
    static Tensor<float> applyConvolutionGridDivision(const Tensor<float>& input, const Tensor<float>& filter, int numThreads);

private:
    static Tensor<float> applyConvolution(const Tensor<float>& input, const Tensor<float>& filter, int numThreads);
    static Tensor<float> apply1DConvolution(const Tensor<float>& input, const Tensor<float>& filter, int numThreads);
};

#endif // FILTERHANDLER_HPP
