#include "FilterHandler.hpp"
#include <opencv2/opencv.hpp>
#include <cmath>
#include <iostream>
#include <omp.h>

Tensor<float> FilterHandler::applyFilter(const Tensor<float>& input, FilterType filterType, int filterSize, int numThreads) {
    omp_set_num_threads(numThreads);
    Tensor<float> filter(1, filterSize, filterSize);

    switch (filterType) {
        case FilterType::MEAN:
            filter = createMeanFilter(filterSize);
            //return applyConvolutionHorizontalDivision(input, filter, numThreads);
        // OR: return applyConvolutionVerticalDivision(input, filter, numThreads);
            return applyConvolutionGridDivision(input, filter, numThreads);
            //break; //if Non-Division
        case FilterType::MEAN1D:
            filter = createMeanFilter1D(filterSize);
            return apply1DConvolution(input, filter, numThreads);
        case FilterType::GAUSSIAN:
            filter = createGaussianFilter(filterSize);
            //return applyConvolutionHorizontalDivision(input, filter, numThreads);
        // OR: return applyConvolutionVerticalDivision(input, filter, numThreads);
            return applyConvolutionGridDivision(input, filter, numThreads);
            //break; //if Non-Division
        case FilterType::GAUSSIAN1D:
            filter = createGaussianFilter1D(filterSize);
            return apply1DConvolution(input, filter, numThreads);
        case FilterType::MEDIAN:
            //return applyMedianFilter(input, filterSize);
            return applyHistogramMedianFilter(input, filterSize);
        case FilterType::MODE:
            return applyModeFilter(input, filterSize);
        default:
            std::cerr << "Unsupported filter type!" << std::endl;
            return input;
    }
    return applyConvolution(input, filter, numThreads);
}

//Flatten matrix function
std::vector<float> flattenImage(const Tensor<float>& image) {
    //std::cout << "Flattening image of size: " << image.height << " x " << image.width << std::endl;
    std::vector<float> flattenedData;
    for (int i = 0; i < image.height; ++i) {
        for (int j = 0; j < image.width; ++j) {
            flattenedData.push_back(image.data[0][i][j]);
        }
    }
    //std::cout << "Flattened image size: " << flattenedData.size() << std::endl;
    return flattenedData;
}

//ReFlatten matrix function
Tensor<float> reflattenToTensor(const std::vector<float>& flattenedData, int height, int width) {
    //std::cout << "Reflatening to Tensor of size: " << height << " x " << width << std::endl;
    Tensor<float> tensor(1, height, width);
    int index = 0;
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            tensor.data[0][i][j] = flattenedData[index++];
        }
    }
    //std::cout << "Reflattened Tensor size: " << tensor.height << " x " << tensor.width << std::endl;
    return tensor;
}

//Normal Mean Filter(3*3 için 1/9)
Tensor<float> FilterHandler::createMeanFilter(int r) {
    int size = 2 * r + 1;
    Tensor<float> filter(1, size, size);

    //#pragma omp parallel safesync //not yet implemented  
    //#pragma omp loop collapse(2) interchange stripe //openmp 6.0
    #pragma omp parallel for collapse(2) //openmp 4.5
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            filter.data[0][i][j] = 1.0f;
        }
    }

    float normalizationFactor = 1.0f / (size * size);
    //#pragma omp parallel safesync //not yet implemented
    //#pragma omp loop collapse(2) interchange stripe //openmp 6.0
    #pragma omp parallel for collapse(2) //openmp 4.5
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            filter.data[0][i][j] *= normalizationFactor;
        }
    }

    return filter;
}
//1D-sequence Mean Filter(aynı size için 1/3)
//matrix flatten. apply. reflatten the 1D
Tensor<float> FilterHandler::createMeanFilter1D(int filterSize) {
    Tensor<float> filter(1, 1, filterSize);

    float normalizationFactor = 1.0f / filterSize;
    #pragma omp parallel for
    for (int i = 0; i < filterSize; ++i) {
        filter.data[0][0][i] = normalizationFactor;
    }

    return filter;
}

Tensor<float> FilterHandler::apply1DConvolution(const Tensor<float>& input, const Tensor<float>& filter, int numThreads) {
    int imageLength = input.height * input.width;
    std::vector<float> flattenedImage = flattenImage(input);

    int filterSize = filter.width;
    int outputHeight = input.height;
    int outputWidth = input.width - filterSize + 1;

    std::vector<float> filteredImage(imageLength, 0.0f);

    #pragma omp parallel for num_threads(numThreads)
    for (int i = 0; i < imageLength; ++i) {
        int start = std::max(0, i - filterSize);
        int end = std::min(imageLength - 1, i + filterSize);

        float sum = 0.0f;
        int count = 0;

        for (int j = start; j <= end; ++j) {
            sum += flattenedImage[j];
            count++;
        }

        filteredImage[i] = sum / count;
    }

    return reflattenToTensor(filteredImage, input.height, input.width);
}

Tensor<float> FilterHandler::createGaussianFilter(int r) {
    float sigma = r / 3.0f;
    int size = 2 * r + 1;
    Tensor<float> filter(1, size, size);
    float mean = size / 2.0f;
    float sum = 0.0f;

    #pragma omp parallel for collapse(2) reduction(+:sum)
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            float x = i - mean;
            float y = j - mean;

            float gaussianValue = std::exp(-(x * x + y * y) / (2 * sigma * sigma));

            filter.data[0][i][j] = gaussianValue;

            sum += gaussianValue;
        }
    }

    #pragma omp parallel for collapse(2) //openmp 4.5
    //#pragma omp simd collapse(2) //openmp 6.0
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            filter.data[0][i][j] /= sum;
        }
    }

    return filter;
}

Tensor<float> FilterHandler::createGaussianFilter1D(int r) {
    float sigma = r / 3.0f;
    int size = 2 * r + 1;
    Tensor<float> filter(1, 1, size);

    float mean = r;
    float sum = 0.0f;

    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < size; ++i) {
        float x = i - mean;
        filter.data[0][0][i] = std::exp(-(x * x) / (2 * sigma * sigma));

        sum += filter.data[0][0][i];
    }

    #pragma omp parallel for //omp 4.5
    //#pragma omp simd
    for (int i = 0; i < size; ++i) {
        filter.data[0][0][i] /= sum;
    }

    return filter;
}

Tensor<float> FilterHandler::applyHistogramMedianFilter(const Tensor<float>& input, int r) {
    int size = 2 * r + 1;
    Tensor<float> output(1, input.height, input.width);

    const int numBins = 256;
    int halfWindow = size * size / 2;

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < input.height; ++i) {
        for (int j = 0; j < input.width; ++j) {
            int histogram[numBins] = {0};

            for (int m = -r; m <= r; ++m) {
                for (int n = -r; n <= r; ++n) {
                    int x = i + m;
                    int y = j + n;

                    if (x >= 0 && x < input.height && y >= 0 && y < input.width) {
                        int bin = static_cast<int>(input.data[0][x][y] * (numBins - 1)); 
                        histogram[bin]++;
                    }
                }
            }

            int count = 0;
            float median = 0.0f;
            for (int bin = 0; bin < numBins; ++bin) {
                count += histogram[bin];
                if (count > halfWindow) {
                    median = static_cast<float>(bin) / (numBins - 1); 
                    break;
                }
            }

            output.data[0][i][j] = median;
        }
    }

    return output;
}

Tensor<float> FilterHandler::applyMedianFilter(const Tensor<float>& input, int r) {
    int size = 2 * r + 1;
    Tensor<float> output(1, input.height, input.width);

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < input.height; ++i) {
        for (int j = 0; j < input.width; ++j) {
            std::vector<float> window;

            //#pragma omp simd collapse(2)
            for (int m = -r; m <= r; ++m) {
                for (int n = -r; n <= r; ++n) {
                    int x = i + m;
                    int y = j + n;

                    if (x >= 0 && x < input.height && y >= 0 && y < input.width) {
                        window.push_back(input.data[0][x][y]);
                    }
                }
            }

            //std::sort(window.begin(), window.end());
            std::nth_element(window.begin(), window.begin() + window.size() / 2, window.end());
            float median = window[window.size() / 2];

            output.data[0][i][j] = median;
        }
    }

    return output;
}

Tensor<float> FilterHandler::applyModeFilter(const Tensor<float>& input, int r) {
    int size = 2 * r + 1;
    int binCount = 256;
    float binSize = 1.0f / binCount;
    Tensor<float> output(1, input.height, input.width);

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < input.height; ++i) {
        for (int j = 0; j < input.width; ++j) {
            //std::vector<float> window;
            int histogram[256] = {0};

            for (int m = -r; m <= r; ++m) {
                for (int n = -r; n <= r; ++n) {
                    int x = i + m;
                    int y = j + n;

                    if (x >= 0 && x < input.height && y >= 0 && y < input.width) {
                        //window.push_back(input.data[0][x][y]);
                        float pixel = input.data[0][x][y]; // Already [0.0, 1.0]
                        int binIndex = std::min(static_cast<int>(pixel * binCount), binCount - 1);
                        histogram[binIndex]++;
                    }
                }
            }

            //std::unordered_map<float, int> frequency;
            //for (float value : window) {
            //    frequency[value]++;
            //}
            int modeBin = 0;
            int maxFrequency = 0;

            for (int b = 0; b < binCount; ++b) {
                if (histogram[b] > maxFrequency) {
                    maxFrequency = histogram[b];
                    modeBin = b;
                }
            }

            //output.data[0][i][j] = mode;
            float modeValue = (modeBin + 0.5f) * binSize; // Center of the bin
            output.data[0][i][j] = modeValue;
        }
    }

    return output;
}


Tensor<float> FilterHandler::loadImage(const std::string& filePath) {
    cv::Mat image = cv::imread(filePath, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Failed to load image: " << filePath << std::endl;
        return Tensor<float>(1, 0, 0);
    }

    Tensor<float> tensor(1, image.rows, image.cols);
    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            tensor.data[0][i][j] = static_cast<float>(image.at<uchar>(i, j)) / 255.0f;  // Normalize to [0, 1]
        }
    }

    return tensor;
}

Tensor<float> FilterHandler::loadRGBImage(const std::string& filePath) {
    cv::Mat image = cv::imread(filePath, cv::IMREAD_COLOR);

    if (image.empty()) {
        std::cerr << "Failed to load image: " << filePath << std::endl;
        return Tensor<float>(0, 0, 0);
    }

    Tensor<float> tensor(3, image.rows, image.cols);

    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            cv::Vec3b color = image.at<cv::Vec3b>(i, j);
            tensor.data[0][i][j] = static_cast<float>(color[2]) / 255.0f;
            tensor.data[1][i][j] = static_cast<float>(color[1]) / 255.0f;
            tensor.data[2][i][j] = static_cast<float>(color[0]) / 255.0f;
        }
    }

    return tensor;
}

Tensor<float> FilterHandler::applyConvolution(const Tensor<float>& input, const Tensor<float>& filter, int numThreads) {
    int filterSize = filter.height;
    int pad = filterSize / 2; // Padding to keep the output size the same

    // Keep the output size the same as input size
    int outputHeight = input.height;
    int outputWidth = input.width;

    Tensor<float> output(1, outputHeight, outputWidth);

    #pragma omp parallel for collapse(2) schedule(dynamic, 16) num_threads(numThreads) //omp 4.5
    for (int i = 0; i < outputHeight; ++i) {
        for (int j = 0; j < outputWidth; ++j) {
            float sum = 0.0f;

            for (int m = 0; m < filterSize; ++m) {
                for (int n = 0; n < filterSize; ++n) {
                    int x = i + m - pad;
                    int y = j + n - pad;

                    if (x >= 0 && x < input.height && y >= 0 && y < input.width) {
                        sum += input.data[0][x][y] * filter.data[0][m][n];
                    }
                }
            }

            output.data[0][i][j] = sum;
        }
    }

    return output;
}

bool FilterHandler::saveImage(const Tensor<float>& image, const std::string& path) {
    //std::cout << "Image channels: " << image.channels << ", height: " << image.height << ", width: " << image.width << std::endl;

    if (image.channels == 1) {
        std::vector<float> flattenedData;
        for (int i = 0; i < image.height; ++i) {
            for (int j = 0; j < image.width; ++j) {
                flattenedData.push_back(image.data[0][i][j]);
            }
        }

        cv::Mat mat(image.height, image.width, CV_32F, flattenedData.data());

        cv::Mat normMat;
        mat.convertTo(normMat, CV_8U, 255.0f);

        return cv::imwrite(path, normMat);
    }

    else if (image.channels == 3) {
        std::vector<cv::Mat> channels(3);

        for (int c = 0; c < 3; ++c) {
            std::vector<float> flattenedData;
            for (int i = 0; i < image.height; ++i) {
                for (int j = 0; j < image.width; ++j) {
                    flattenedData.push_back(image.data[c][i][j]);
                }
            }

            channels[c] = cv::Mat(image.height, image.width, CV_32F, flattenedData.data());


            cv::normalize(channels[c], channels[c], 0, 255, cv::NORM_MINMAX);
            channels[c].convertTo(channels[c], CV_8U);  // Convert to 8-bit
        }

        cv::Mat rgbImage;
        cv::merge(channels, rgbImage);

        return cv::imwrite(path, rgbImage);
    }

    std::cerr << "Unsupported image format!" << std::endl;
    return false;
}

bool FilterHandler::saveMultipleChannelsImage(const Tensor<float>& image, const std::string& basePath) {
    //std::cout << "Saving image with multiple channels: " << image.channels << ", height: " << image.height << ", width: " << image.width << std::endl;

    for (int c = 0; c < image.channels; ++c) {
        std::vector<float> flattenedData;
        for (int i = 0; i < image.height; ++i) {
            for (int j = 0; j < image.width; ++j) {
                flattenedData.push_back(image.data[c][i][j]);
            }
        }

        cv::Mat mat(image.height, image.width, CV_32F, flattenedData.data());

        cv::Mat normMat;
        mat.convertTo(normMat, CV_8U, 255.0f);

        std::string channelPath = basePath + "_channel_" + std::to_string(c) + ".png";

        if (!cv::imwrite(channelPath, normMat)) {
            std::cerr << "Failed to save channel " << c << "!" << std::endl;
            return false;
        }
    }

    return true;
}

bool FilterHandler::saveCombinedImage(Tensor<float>& image, const std::string& basePath) {
    std::vector<cv::Mat> channels;

    for (int c = 0; c < image.channels; ++c) {
        std::vector<float> flattenedData;

        for (int i = 0; i < image.height; ++i) {
            for (int j = 0; j < image.width; ++j) {
                flattenedData.push_back(image.data[c][i][j]);
            }
        }

        cv::Mat mat(image.height, image.width, CV_32F, flattenedData.data());

        cv::Mat normMat;
        cv::normalize(mat, normMat, 0, 255, cv::NORM_MINMAX);
        normMat.convertTo(normMat, CV_8U);

        channels.push_back(normMat);
    }

    cv::Mat combinedImage;

    cv::vconcat(channels, combinedImage);

    //std::cout << "Saving combined image to: " << basePath << std::endl;

    if (cv::imwrite(basePath, combinedImage)) {
        std::cout << "Combined image saved successfully to " << basePath << "!" << std::endl;
        return true;
    } else {
        std::cerr << "Failed to save combined image!" << std::endl;
        return false;
    }
}

// ============================================================================
// HORIZONTAL DIVISION: Each thread processes horizontal strips of the image
// ============================================================================
Tensor<float> FilterHandler::applyConvolutionHorizontalDivision(
    const Tensor<float>& input, 
    const Tensor<float>& filter, 
    int numThreads) 
{
    int filterSize = filter.height;
    int pad = filterSize / 2;
    
    int outputHeight = input.height;
    int outputWidth = input.width;
    
    Tensor<float> output(1, outputHeight, outputWidth);
    
    // Divide image into horizontal strips
    int rowsPerThread = (outputHeight + numThreads - 1) / numThreads;
    
    #pragma omp parallel num_threads(numThreads)
    {
        int threadId = omp_get_thread_num();
        int startRow = threadId * rowsPerThread;
        int endRow = std::min(startRow + rowsPerThread, outputHeight);
        
        // Each thread processes its assigned horizontal strip
        for (int i = startRow; i < endRow; ++i) {
            for (int j = 0; j < outputWidth; ++j) {
                float sum = 0.0f;
                
                for (int m = 0; m < filterSize; ++m) {
                    for (int n = 0; n < filterSize; ++n) {
                        int x = i + m - pad;
                        int y = j + n - pad;
                        
                        if (x >= 0 && x < input.height && y >= 0 && y < input.width) {
                            sum += input.data[0][x][y] * filter.data[0][m][n];
                        }
                    }
                }
                
                output.data[0][i][j] = sum;
            }
        }
    }
    
    return output;
}

// ============================================================================
// VERTICAL DIVISION: Each thread processes vertical strips of the image
// ============================================================================
Tensor<float> FilterHandler::applyConvolutionVerticalDivision(
    const Tensor<float>& input, 
    const Tensor<float>& filter, 
    int numThreads) 
{
    int filterSize = filter.height;
    int pad = filterSize / 2;
    
    int outputHeight = input.height;
    int outputWidth = input.width;
    
    Tensor<float> output(1, outputHeight, outputWidth);
    
    // Divide image into vertical strips
    int colsPerThread = (outputWidth + numThreads - 1) / numThreads;
    
    #pragma omp parallel num_threads(numThreads)
    {
        int threadId = omp_get_thread_num();
        int startCol = threadId * colsPerThread;
        int endCol = std::min(startCol + colsPerThread, outputWidth);
        
        // Each thread processes its assigned vertical strip
        for (int i = 0; i < outputHeight; ++i) {
            for (int j = startCol; j < endCol; ++j) {
                float sum = 0.0f;
                
                for (int m = 0; m < filterSize; ++m) {
                    for (int n = 0; n < filterSize; ++n) {
                        int x = i + m - pad;
                        int y = j + n - pad;
                        
                        if (x >= 0 && x < input.height && y >= 0 && y < input.width) {
                            sum += input.data[0][x][y] * filter.data[0][m][n];
                        }
                    }
                }
                
                output.data[0][i][j] = sum;
            }
        }
    }
    
    return output;
}

// ============================================================================
// GRID DIVISION (NxN): Each thread processes a tile in a 2D grid
// For 4 threads on 192x192 image: creates 2x2 grid with 96x96 tiles each
// ============================================================================
Tensor<float> FilterHandler::applyConvolutionGridDivision(
    const Tensor<float>& input, 
    const Tensor<float>& filter, 
    int numThreads) 
{
    int filterSize = filter.height;
    int pad = filterSize / 2;
    
    int outputHeight = input.height;
    int outputWidth = input.width;
    
    Tensor<float> output(1, outputHeight, outputWidth);
    
    // Calculate grid dimensions (as square as possible)
    int gridRows = static_cast<int>(std::sqrt(numThreads));
    int gridCols = (numThreads + gridRows - 1) / gridRows;
    
    // Adjust if needed to use all threads
    while (gridRows * gridCols < numThreads && gridCols < outputWidth) {
        gridCols++;
    }
    
    int rowsPerTile = (outputHeight + gridRows - 1) / gridRows;
    int colsPerTile = (outputWidth + gridCols - 1) / gridCols;
    
    #pragma omp parallel num_threads(numThreads)
    {
        int threadId = omp_get_thread_num();
        
        // Calculate which tile this thread handles
        int tileRow = threadId / gridCols;
        int tileCol = threadId % gridCols;
        
        // Skip if thread ID exceeds grid dimensions
        if (tileRow >= gridRows) {
            // Do nothing - extra threads remain idle
        } else {
            // Calculate boundaries for this tile
            int startRow = tileRow * rowsPerTile;
            int endRow = std::min(startRow + rowsPerTile, outputHeight);
            int startCol = tileCol * colsPerTile;
            int endCol = std::min(startCol + colsPerTile, outputWidth);
            
            // Each thread processes its assigned tile
            for (int i = startRow; i < endRow; ++i) {
                for (int j = startCol; j < endCol; ++j) {
                    float sum = 0.0f;
                    
                    for (int m = 0; m < filterSize; ++m) {
                        for (int n = 0; n < filterSize; ++n) {
                            int x = i + m - pad;
                            int y = j + n - pad;
                            
                            if (x >= 0 && x < input.height && y >= 0 && y < input.width) {
                                sum += input.data[0][x][y] * filter.data[0][m][n];
                            }
                        }
                    }
                    
                    output.data[0][i][j] = sum;
                }
            }
        }
    }
    
    return output;
}

std::string FilterHandler::getFilterName(FilterType filter) {
    switch (filter) {
        case FilterType::MEAN: return "mean";
        case FilterType::MEAN1D: return "mean1d";
        case FilterType::GAUSSIAN: return "gaussian";
        case FilterType::MEDIAN: return "median";
        case FilterType::MODE: return "mode";
        default: return "unknown";
    }
}
