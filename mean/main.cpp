#include <iostream>
#include <string>
#include <omp.h>
#include <chrono>
#include "../common/FilterHandler.hpp"

int main(int argc, char* argv[]) {

    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <filter_radius> <num_threads>\n";
        return 1;
    }
    // image path (change to your image path)
    std::string imagePath = "/home/abdulkadir/İndirilenler/a5000-kme_0204.tif";

    int filterRadius = std::stoi(argv[1]);
    int numThreads = std::stoi(argv[2]);
    omp_set_num_threads(numThreads);

    // Load the image
    Tensor<float> inputImage = FilterHandler::loadImage(imagePath);
    if (inputImage.height == 0) {
        std::cerr << "Error loading image!\n";
        return 1;
    }

    auto start = std::chrono::high_resolution_clock::now();

    Tensor<float> filteredImage = FilterHandler::applyFilter(inputImage, FilterType::MEAN, filterRadius, numThreads);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::string outputPath = imagePath.substr(0, imagePath.rfind('.')) + "_mean_r" + std::to_string(filterRadius) + ".png";

    if (FilterHandler::saveImage(filteredImage, outputPath)) {
        std::cout << "Filtered image saved to: " << outputPath << "\n";
    } else {
        std::cerr << "Error saving filtered image!\n";
    }

    std::cout << "Filtering completed in " << elapsed.count() << " seconds.\n";

    return 0;
}

//Performance cores
//export OMP_PLACES="{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
//export OMP_NUM_THREADS=16

//Unbound
//unset OMP_PLACES

//Efficiency cores
//export OMP_PLACES="{16,17,18,19,20,21,22,23,24,25,26,27}"
//export OMP_NUM_THREADS=12