#include <stdio.h>
#include <stdlib.h>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "kernels.h"
#include "cpu_kernels.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/core/cuda.hpp>

extern "C" {
    #include "stb_image.h"
    #include "stb_image_write.h"
    #include "timing.h"
}

int main(int argc, char** argv ) {

    if (argc != 2)
    {
        printf("usage: ./project <Image_Path>\n");
        return -1;
    }

    double now, then;
    double scost, pcost;

    double kernel_now, kernel_then;
    double gpu_kernel_time,

    char *input_image_path = argv[1];

    printf("Reading image %s\n", input_image_path);

    // Load the image using stb_image
    int width, height, channels;
    unsigned char *img = stbi_load(input_image_path, &width, &height, &channels, 3);

    if (!img) {
        printf("Failed to load image: %s\n", stbi_failure_reason());
        return 1;
    }

    printf("Width: %d Height: %d\n", width, height);

    // ---------------- CPU PIPELINE ----------------
    float *cpu_hsv_pixels = (float*)malloc(width * height * 3 * sizeof(float));
    unsigned char *cpu_red_pixels = (unsigned char*)malloc(width * height * sizeof(unsigned char));
    unsigned char *cpu_smoothed_pixels = (unsigned char*)malloc(width * height * sizeof(unsigned char));

    if (!cpu_hsv_pixels || !cpu_red_pixels || !cpu_smoothed_pixels) {
        printf("Failed to allocate CPU memory\n");
        free(img);
        return 1;
    }

    then = currentTime();

    rgbToHsvCPU(img, cpu_hsv_pixels, height, width);
    filterRedCPU(cpu_hsv_pixels, cpu_red_pixels, height, width);
    gaussianSmoothCPU(cpu_red_pixels, cpu_smoothed_pixels, height, width);

    //Use OpenCV library to run Canny Edge Detection
    cv::Mat cpu_smoothed_mat(height, width, CV_8UC1, cpu_smoothed_pixels);

    cv::Mat cpu_canny_edges;
    cv::Canny(cpu_smoothed_mat, cpu_canny_edges, 50.0, 150.0);

    now = currentTime();
    scost = (now - then) * 1000.0;

    stbi_write_png("cpu_smoothed_output.png", width, height, 1, cpu_smoothed_pixels, width);
    cv::imwrite("cpu_canny_edges_output.png", cpu_canny_edges);
    printf("Images written to cpu_smoothed_output.png and cpu_canny_edges_output.png\n");

    // ---------------- GPU PIPELINE ----------------
    then = currentTime();

    unsigned char *device_rgb_pixels;
    float *device_hsv_pixels;
    unsigned char *device_output_pixels;
    unsigned char *device_smoothed_pixels;

    int charByteSize = width * height * sizeof(unsigned char) * 3;
    int floatByteSize = width * height * sizeof(float) * 3;

    cudaMalloc(&device_rgb_pixels, charByteSize);
    cudaMalloc(&device_hsv_pixels, floatByteSize);
    cudaMalloc(&device_output_pixels, width * height * sizeof(unsigned char));
    cudaMalloc(&device_smoothed_pixels, width * height * sizeof(unsigned char));

    cudaMemcpy(device_rgb_pixels, img, charByteSize, cudaMemcpyHostToDevice);

    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    //Timing for only kernel execution
    kernel_then = currentTime();

    rgbToHsvKernel<<<grid, block>>>(device_rgb_pixels, device_hsv_pixels, height, width);
    cudaDeviceSynchronize();

    filterRed<<<grid, block>>>(device_hsv_pixels, device_output_pixels, height, width);
    cudaDeviceSynchronize();

    gaussianSmoothKernel<<<grid, block>>>(device_output_pixels, device_smoothed_pixels, height, width);
    cudaDeviceSynchronize();

    cv::Mat cpu_smoothed_temp(height, width, CV_8UC1);
    cudaMemcpy(cpu_smoothed_temp.data, device_smoothed_pixels, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cv::cuda::GpuMat gpu_smoothed;
    gpu_smoothed.upload(cpu_smoothed_temp);
    cv::cuda::GpuMat gpu_edges;

    auto canny = cv::cuda::createCannyEdgeDetector(50.0, 150.0);
    canny->detect(gpu_smoothed, gpu_edges);

    kernel_now = currentTime();
    gpu_kernel_time = ((kernel_now - kernel_then) * 1000.0);
    
    //Download edges and saves them
    cv::Mat cpu_edges;
    gpu_edges.download(cpu_edges);

    unsigned char *smoothed_pixels = (unsigned char*)malloc(width * height * sizeof(unsigned char));
    cudaMemcpy(smoothed_pixels, device_smoothed_pixels, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    now = currentTime();
    pcost = (now - then) * 1000.0;

    cv::imwrite("gpu_canny_edges_output.png", cpu_edges);
    stbi_write_png("gpu_smoothed_output.png", width, height, 1, smoothed_pixels, width);
    printf("Images written to gpu_smoothed_output.png and canny_edges_output.png\n\n");

    printf("CPU Time taken: %f ms\n", scost);
    printf("GPU Time taken: %f ms\n", pcost);
    if (scost > 0.0) {
        printf("Speedup: %.2fx\n\n", scost / pcost);
    }

    printf("ONLY GPU Kernel execution time: %f ms\n", gpu_kernel_time);
    if (scost > 0.0) {
        printf("Kernel ONLY Speedup: %.2fx\n\n", scost / gpu_kernel_time);
    }

    cudaFree(device_hsv_pixels);
    cudaFree(device_rgb_pixels);
    cudaFree(device_output_pixels);
    cudaFree(device_smoothed_pixels);

    free(cpu_hsv_pixels);
    free(cpu_red_pixels);
    free(cpu_smoothed_pixels);
    free(smoothed_pixels);
    free(img);

    return 0;
}