#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <chrono>
#include <cuda_runtime.h>

// CUDA error checking macro
#define CUDA_CHECK(call)     do {         cudaError_t err = call;         if (err != cudaSuccess) {             std::cerr << "CUDA Error: " << cudaGetErrorString(err)                       << " at line " << __LINE__ << std::endl;             exit(EXIT_FAILURE);         }     } while (0)

// Struct to hold PPM input image metadata and raw RGB data
struct PPMImage {
    int width;
    int height;
    int max_val;
    std::vector<unsigned char> data;
};

// Helper to read P6 PPM RGB image
PPMImage readPPM(const std::string& filename) {
    std::ifstream infile(filename, std::ios::binary);
    if (!infile.is_open()) {
        std::cerr << "Error: Could not open " << filename << " for reading!" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string format;
    infile >> format;
    if (format != "P6") {
        std::cerr << "Error: Only binary PPM (P6) format is supported! Got: " << format << std::endl;
        exit(EXIT_FAILURE);
    }

    // Skip whitespace and comments
    char ch;
    infile >> ch;
    while (ch == '#') {
        std::string comment;
        std::getline(infile, comment);
        infile >> ch;
    }
    infile.unget();

    PPMImage img;
    infile >> img.width >> img.height >> img.max_val;
    infile.get(); // Consume newline

    int num_pixels = img.width * img.height;
    img.data.resize(num_pixels * 3);
    infile.read(reinterpret_cast<char*>(img.data.data()), num_pixels * 3);
    infile.close();

    return img;
}

// Helper to write P5 PGM grayscale image
void writePGM(const std::string& filename, const std::vector<unsigned char>& data, int width, int height) {
    std::ofstream outfile(filename, std::ios::binary);
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open " << filename << " for writing!" << std::endl;
        exit(EXIT_FAILURE);
    }

    outfile << "P5\n" << width << " " << height << "\n255\n";
    outfile.write(reinterpret_cast<const char*>(data.data()), width * height);
    outfile.close();
}

// CUDA Kernel for converting RGB to Grayscale
__global__ void rgbToGrayscaleKernel(const unsigned char* d_rgb, unsigned char* d_gray, int width, int height) {
    // 2D grid/block index mapping
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Handle boundaries correctly
    if (x < width && y < height) {
        int pixel_idx = y * width + x;
        int rgb_offset = pixel_idx * 3;

        unsigned char r = d_rgb[rgb_offset];
        unsigned char g = d_rgb[rgb_offset + 1];
        unsigned char b = d_rgb[rgb_offset + 2];

        // Standard grayscale luminance formula
        d_gray[pixel_idx] = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

int main(int argc, char* argv[]) {
    std::string input_dir = "input";
    std::string output_dir = "output";
    int num_images = 100;

    // Parse simple command-line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--input" && i + 1 < argc) {
            input_dir = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (arg == "--images" && i + 1 < argc) {
            num_images = std::stoi(argv[++i]);
        }
    }

    // Retrieve GPU Device properties
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "==============================================" << std::endl;
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "==============================================" << std::endl;

    // Timers
    float total_kernel_time_ms = 0.0f;
    auto total_cpu_start = std::chrono::high_resolution_clock::now();

    int width = 0, height = 0;
    long long total_pixels_processed = 0;

    for (int i = 1; i <= num_images; ++i) {
        char filename_buf[128];
        snprintf(filename_buf, sizeof(filename_buf), "image_%03d.ppm", i);
        std::string input_path = input_dir + "/" + filename_buf;

        snprintf(filename_buf, sizeof(filename_buf), "image_%03d.pgm", i);
        std::string output_path = output_dir + "/" + filename_buf;

        // 1. Load image (CPU)
        PPMImage host_input = readPPM(input_path);
        width = host_input.width;
        height = host_input.height;
        total_pixels_processed += (width * height);

        int rgb_size = width * height * 3 * sizeof(unsigned char);
        int gray_size = width * height * sizeof(unsigned char);

        std::vector<unsigned char> host_output(width * height);

        // 2. Allocate device memory (GPU)
        unsigned char* d_rgb = nullptr;
        unsigned char* d_gray = nullptr;
        CUDA_CHECK(cudaMalloc(&d_rgb, rgb_size));
        CUDA_CHECK(cudaMalloc(&d_gray, gray_size));

        // 3. Copy Host to Device (CPU -> GPU)
        CUDA_CHECK(cudaMemcpy(d_rgb, host_input.data.data(), rgb_size, cudaMemcpyHostToDevice));

        // 4. Configure CUDA grid & block size
        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

        // 5. Launch CUDA Grayscale Kernel with timing events
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));
        rgbToGrayscaleKernel<<<grid, block>>>(d_rgb, d_gray, width, height);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaDeviceSynchronize());

        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        total_kernel_time_ms += milliseconds;

        // Cleanup events
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        // 6. Copy Device to Host (GPU -> CPU)
        CUDA_CHECK(cudaMemcpy(host_output.data(), d_gray, gray_size, cudaMemcpyDeviceToHost));

        // 7. Save Grayscale image (CPU)
        writePGM(output_path, host_output, width, height);

        // Cleanup GPU arrays
        CUDA_CHECK(cudaFree(d_rgb));
        CUDA_CHECK(cudaFree(d_gray));
    }

    auto total_cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> total_elapsed = total_cpu_end - total_cpu_start;

    // Calculate parameters for demonstration
    dim3 block_demo(16, 16);
    dim3 grid_demo((width + block_demo.x - 1) / block_demo.x, (height + block_demo.y - 1) / block_demo.y);

    std::cout << "Images processed:           " << num_images << std::endl;
    std::cout << "Resolution:                 " << width << " x " << height << std::endl;
    std::cout << "Total pixels:               " << total_pixels_processed << std::endl;
    std::cout << "CUDA block size:            " << block_demo.x << " x " << block_demo.y << std::endl;
    std::cout << "CUDA grid size:             " << grid_demo.x << " x " << grid_demo.y << std::endl;
    std::cout << "GPU kernel execution time:  " << total_kernel_time_ms << " ms" << std::endl;
    std::cout << "Total processing time:      " << total_elapsed.count() << " ms" << std::endl;
    std::cout << "Processing completed successfully." << std::endl;

    return 0;
}