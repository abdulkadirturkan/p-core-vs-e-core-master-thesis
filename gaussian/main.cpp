#include <iostream>
#include <string>
#include <chrono>
#include <omp.h>
#include "../common/FilterHandler.hpp"

int main(int argc, char* argv[]) {

    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <filter_radius> <num_threads>\n";
        return 1;
    }
    // Hardcoded image path (change to your image path)
    std::string imagePath = "/home/abdulkadir/İndirilenler/a5000-kme_0204.tif";

    int filterRadius = std::stoi(argv[1]);
    int numThreads = std::stoi(argv[2]);
    omp_set_num_threads(numThreads);

    //std::cout << "Applying Mean Filter...\n";
    //std::cout << "Enter filter radius (e.g. 3 for 7x7, 5 for 11x11): ";
    //int filterRadius;
    //std::cin >> filterRadius;

    //int numThreads;
    //std::cout << "Enter the number of threads (e.g., 1 for threads): ";
    //std::cin >> numThreads;
    //omp_set_num_threads(numThreads);

    // Load the image
    Tensor<float> inputImage = FilterHandler::loadImage(imagePath);
    if (inputImage.height == 0) {
        std::cerr << "Error loading image!\n";
        return 1;
    }

    auto start = std::chrono::high_resolution_clock::now();

    Tensor<float> filteredImage = FilterHandler::applyFilter(inputImage, FilterType::GAUSSIAN, filterRadius, numThreads);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::string outputPath = imagePath.substr(0, imagePath.rfind('.')) + "_gaussian_r" + std::to_string(filterRadius) + ".png";

    if (FilterHandler::saveImage(filteredImage, outputPath)) {
        std::cout << "Filtered image saved to: " << outputPath << "\n";
    } else {
        std::cerr << "Error saving filtered image!\n";
    }

    std::cout << "Filtering completed in " << elapsed.count() << " seconds.\n";

    return 0;
}

